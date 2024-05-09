use crate::optimizer::{Momentum, SGD};
use crate::two_layer_net::TwoLayerNet;
use crate::types::{NNFloat, NNMatrix};
use mnist::*;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use std::cmp::max;

mod functions;
mod layers;
mod optimizer;
mod two_layer_net;
mod types;

fn main() {
    let trn_len = 60_000;
    let tst_len = 10_000;
    let input_size = 784;
    let NormalizedMnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_one_hot()
        .training_set_length(trn_len)
        .test_set_length(tst_len)
        .finalize()
        .normalize();

    let mut network = TwoLayerNet::new(input_size, 50, 10, 0.01);

    let x_train = Array2::from_shape_vec((trn_len as usize, input_size), trn_img)
        .expect("Error converting images to Array2 struct");

    let t_train = Array2::from_shape_vec((trn_len as usize, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as NNFloat);

    let x_test = Array2::from_shape_vec((tst_len as usize, input_size), tst_img)
        .expect("Error converting images to Array2 struct");

    let t_test = Array2::from_shape_vec((tst_len as usize, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as NNFloat);

    let iters_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    // let learning_rate = 0.1;

    let mut train_loss_list: Vec<NNFloat> = vec![];
    let mut train_acc_list: Vec<NNFloat> = vec![];
    let mut test_acc_list: Vec<NNFloat> = vec![];
    let iter_per_epoch = max(train_size / batch_size, 1);
    let induce = Array::range(0.0, train_size as NNFloat, 1.0).mapv(|a| a as usize);
    // let optimizer = SGD::new(learning_rate);
    let mut optimizer = Momentum::default();

    for i in 0..iters_num {
        let batch_mask = induce
            .sample_axis(Axis(0), batch_size, SamplingStrategy::WithoutReplacement)
            .to_vec();
        let mut x_batch_vec: Vec<NNFloat> = vec![];
        let mut t_batch_vec: Vec<NNFloat> = vec![];
        for i in batch_mask {
            x_batch_vec.append(&mut x_train.index_axis(Axis(0), i).to_vec());
            t_batch_vec.append(&mut t_train.index_axis(Axis(0), i).to_vec());
        }
        let x_batch: NNMatrix =
            Array::from_shape_vec((batch_size, input_size), x_batch_vec).expect("");
        let t_batch: NNMatrix = Array::from_shape_vec((batch_size, 10), t_batch_vec).expect("");
        network.train(&x_batch, &t_batch, &mut optimizer);

        let loss = network.loss(&x_batch, &t_batch);
        train_loss_list.push(loss);

        if i % iter_per_epoch == 0 {
            println!("loss: {}", loss);
            let train_acc = network.accuary(&x_train, &t_train);
            let test_acc = network.accuary(&x_test, &t_test);
            train_acc_list.push(train_acc);
            test_acc_list.push(test_acc);
            println!("trn: {}, tst: {}", train_acc, test_acc);
        }
    }
}
