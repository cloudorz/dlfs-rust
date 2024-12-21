use crate::optimizer::{SGD, Momentum};
use crate::two_layer_net::TwoLayerNet;
use crate::types::{NNFloat, NNMatrix};
use mnist::*;
use ndarray::{Array, Array2, Axis};
use ndarray_rand::{RandomExt, SamplingStrategy};
use simple_conv_net::SimpleConvNet;
use std::cmp::max;

mod functions;
mod layers;
mod optimizer;
mod two_layer_net;
mod types;
mod simple_conv_net;

fn main() {
    simple_cnn();
    // affine_net();
}

fn simple_cnn() {
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

    let trn_lbl_: Vec<NNFloat> = trn_lbl.iter().map(|x| *x as NNFloat).collect();
    // let mut network = TwoLayerNet::new(input_size, 50, 10, 0.01);
    let mut network = SimpleConvNet::new([1, 28, 28], [30, 5, 0, 1], 100, 10, 0.01);

    let trn_img_ = trn_img.clone();
    let x_train = Array::from_shape_vec((trn_len as usize, 1, 28, 28), trn_img)
        .expect("Error converting images to Array2 struct");

    let t_train = Array::from_shape_vec((trn_len as usize, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as NNFloat);

    let x_test = Array::from_shape_vec((tst_len as usize, 1, 28, 28), tst_img)
        .expect("Error converting images to Array2 struct");

    let t_test = Array::from_shape_vec((tst_len as usize, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as NNFloat);

    let iters_num = 10000;
    let train_size = x_train.shape()[0];
    let batch_size = 100;
    let learning_rate = 0.1;

    let mut train_loss_list: Vec<NNFloat> = vec![];
    let mut train_acc_list: Vec<NNFloat> = vec![];
    let mut test_acc_list: Vec<NNFloat> = vec![];
    let iter_per_epoch = max(train_size / batch_size, 1);
    let induce = Array::range(0.0, train_size as NNFloat, 1.0).mapv(|a| a as usize);
    let mut optimizer = SGD::new(learning_rate);
    // let mut optimizer = Momentum::default();

    for i in 0..iters_num {
        let batch_mask = induce
            .sample_axis(Axis(0), batch_size, SamplingStrategy::WithoutReplacement)
            .to_vec();
        let mut x_batch_vec: Vec<NNFloat> = vec![];
        let mut t_batch_vec: Vec<NNFloat> = vec![];
        for i in batch_mask {
            x_batch_vec.extend(&trn_img_[i*input_size..(i*input_size+input_size)]);
            t_batch_vec.extend(&trn_lbl_[i*10..(i*10+10)]);
        }
        let x_batch =
            Array::from_shape_vec((batch_size, 1, 28, 28), x_batch_vec).expect("");
        let t_batch = Array::from_shape_vec((batch_size, 10), t_batch_vec).expect("");
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

    // Plot training and test accuracy
    visualize_chart("Training/Test Accuracy", "accuray.png", (iters_num/iter_per_epoch) as f32, 1f32, vec![("Training Accuracy", RED.filled(), train_acc_list), ("Test Accuracy", BLUE.filled(), test_acc_list)]);
    visualize_chart("Training Loss", "train_loss.png", iters_num as NNFloat, 2.5f32, vec![("Loss", RED.filled(), train_loss_list)]);
}


fn affine_net() {
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
    let learning_rate = 0.1;

    let mut train_loss_list: Vec<NNFloat> = vec![];
    let mut train_acc_list: Vec<NNFloat> = vec![];
    let mut test_acc_list: Vec<NNFloat> = vec![];
    let iter_per_epoch = max(train_size / batch_size, 1);
    let induce = Array::range(0.0, train_size as NNFloat, 1.0).mapv(|a| a as usize);
    let mut optimizer = SGD::new(learning_rate);
    // let mut optimizer = Momentum::default();

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

    // Plot training and test accuracy
    visualize_chart("Training/Test Accuracy", "accuray.png", (iters_num/iter_per_epoch) as f32, 1f32, vec![("Training Accuracy", RED.filled(), train_acc_list), ("Test Accuracy", BLUE.filled(), test_acc_list)]);
    visualize_chart("Training Loss", "train_loss.png", iters_num as NNFloat, 2.5f32, vec![("Loss", RED.filled(), train_loss_list)]);
}



use plotters::prelude::*;
fn visualize_chart(chart_name: &str, image_name: &str, x_max: NNFloat, y_max: NNFloat, data: Vec<(&str, ShapeStyle, Vec<NNFloat>)>) {
    let root = BitMapBackend::new(image_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(chart_name, ("sans-serif", 32).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..x_max, 0f32..y_max)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    for (label, style, list) in data {
        let style = style.clone();
        chart
            .draw_series(LineSeries::new(
                list.iter().enumerate().map(|(x, y)| (x as f32, *y)),
                style,
            ))
            .unwrap()
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], style));
    }
}