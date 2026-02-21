use bullet_lib::{
    game::{
        inputs::{ChessBucketsMirrored, get_num_buckets},
        outputs::MaterialCount,
    },
    nn::{
        InitSettings, Shape,
        optimiser::{AdamW, AdamWParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder},
};

use bullet_lib::value::loader::SfBinpackLoader;
use sfbinpack::TrainingDataEntry;
use sfbinpack::chess::piecetype::PieceType;


fn main() {
    let l1_size = 1024;
    let initial_lr = 0.001;
    let final_lr = initial_lr * 0.3 * 0.3 * 0.3;
    let superbatches = 100;
    const NUM_OUTPUT_BUCKETS: usize = 8;

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .output_buckets(MaterialCount::<NUM_OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").transpose().round().quantise::<i8>(64),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w").transpose(),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w").transpose(),
            SavedFormat::id("l3b"),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            // input layer weights
            let l0 = builder.new_affine("l0", 768, l1_size);

            // layerstack weights
            let l1 = builder.new_affine("l1", l1_size, NUM_OUTPUT_BUCKETS * 16);
            let l2 = builder.new_affine("l2", 16, NUM_OUTPUT_BUCKETS * 32);
            let l3 = builder.new_affine("l3", 32, NUM_OUTPUT_BUCKETS);

            // inference
            let stm_hidden = l0.forward(stm_inputs).crelu().pairwise_mul();
            let ntm_hidden = l0.forward(ntm_inputs).crelu().pairwise_mul();
            let hl1 = stm_hidden.concat(ntm_hidden);
            let hl2 = l1.forward(hl1).select(output_buckets).crelu();
            let hl3 = l2.forward(hl2).select(output_buckets).crelu();
            l3.forward(hl3).select(output_buckets)
        });

    trainer.optimiser.set_params(AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    });

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.7 },
        lr_scheduler: lr::CosineDecayLR { initial_lr, final_lr, final_superbatch: superbatches },
        save_rate: 100,
    };

    let settings = LocalSettings { threads: 8, test_set: None, output_directory: "checkpoints", batch_queue_size: 64 };

    let dataloader = {
        let file_path = "/workspace/data.binpack";
        let buffer_size_mb = 4096;
        let threads = 8;
        fn filter(entry: &TrainingDataEntry) -> bool {
                !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 32000
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }
        SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };

    trainer.run(&schedule, &settings, &dataloader);

    for fen in [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ] {
        let eval = trainer.eval(fen);
        println!("FEN: {fen}");
        println!("EVAL: {}", 400.0 * eval);
    }
}
