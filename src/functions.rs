use crate::types::*;
use ndarray::prelude::*;
use std::iter::zip;

fn exp<D: Dimension>(x: Array<NNFloat, D>) -> Array<NNFloat, D> {
    x.mapv_into(|value| value.exp())
}

pub fn sigmoid<D: Dimension>(x: &Array<NNFloat, D>) -> Array<NNFloat, D> {
    1.0 / (1.0 + exp(-x))
}

pub fn softmax(x: &NNMatrix) -> NNMatrix {
    // P.S: use `broadcast()` + reshape to avoid using `t()` twice ?
    let x = x.t().to_owned();
    let x_max = x.fold_axis(Axis(0), 0.0, |acc, v| if *acc < *v { *v } else { *acc });
    let x = exp(x - x_max);
    let x_sum = x.sum_axis(Axis(0));
    let y = x / x_sum;

    y.t().to_owned()
}

pub fn cross_entropy_error(y: &NNMatrix, t: &NNMatrix) -> NNFloat {
    let batch_size = y.shape()[0];
    let k = if y.len() == t.len() {
        // one hot vector
        t * y.mapv(|v| (v + 1e-7).ln())
    } else {
        let mut d = Array::<NNFloat, _>::zeros(t.raw_dim());
        for (i, arr) in t.axis_iter(Axis(0)).enumerate() {
            for (j, pos) in arr.iter().enumerate() {
                d[[i, j]] = (y[[i, *pos as usize]] + 1e-7).ln();
            }
        }
        d
    };
    -k.sum() / (batch_size as NNFloat)
}

pub fn accuary(x: &NNMatrix, t: &NNMatrix) -> NNFloat {
    let t = t.map_axis(Axis(1), |arr| {
        if arr.len() == 1 {
            arr[0] as usize
        } else {
            arr.iter().position(|value| *value == 1.0).unwrap()
        }
    });
    let size = x.shape()[0];
    let x = x.map_axis(Axis(1), |arr| {
        let max_v = arr
            .iter()
            .max_by(|x1, x2| x1.partial_cmp(x2).unwrap())
            .unwrap();
        arr.iter().position(|v| *v == *max_v).unwrap()
    });
    let match_num = zip(x, t).fold(0.0, |acc, (y1, t1)| if y1 == t1 { acc + 1.0 } else { acc });

    match_num / size as NNFloat
}

pub fn im2col(
    input_data: &Array4<NNFloat>,
    filter_h: usize,
    filter_w: usize,
    stride: usize,
    pad: usize,
) -> NNMatrix {
    let shape = input_data.shape();
    let input_data_number = shape[0];
    let channel_count = shape[1];
    let matrix_height = shape[2];
    let matrix_width = shape[3];
    let out_h = (matrix_height + 2 * pad - filter_h) / stride + 1;
    let out_w = (matrix_width + 2 * pad - filter_w) / stride + 1;

    let mut col = NNMatrix::zeros((
        input_data_number * out_h * out_w,
        filter_h * filter_w * channel_count,
    ));
    for row in 0..col.shape()[0] {
        let input_data_index = row / (out_w * out_h);
        let m = row % (out_w * out_h);
        let input_data_height_index = m / out_h * stride;
        let input_data_width_index = m % out_h * stride;
        for column in 0..col.shape()[1] {
            let channel_index = column / (filter_w * filter_h);
            let m = column % (filter_w * filter_h);
            let filter_height_index = m / filter_h;
            let filter_width_index = m % filter_h;
            let height_index = input_data_height_index + filter_height_index;
            let width_index = input_data_width_index + filter_width_index;
            let padded_width = matrix_width + 2 * pad;
            let padded_height = matrix_height + 2 * pad;

            if pad > 0
                && (width_index < pad
                    || (width_index < padded_width && width_index >= padded_width - pad)
                    || height_index < pad
                    || (height_index < padded_height && height_index >= padded_height - pad))
            {
                // padding
                col[[row, column]] = 0.0;
            } else {
                col[[row, column]] = input_data[[
                    input_data_index,
                    channel_index,
                    height_index - pad,
                    width_index - pad,
                ]];
            }
        }
    }

    col
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sigmoid() {
        let x = array![-1.0, 1.0, 2.0];
        let x_results = array![0.26894142, 0.731_058_6, 0.880797];
        assert_eq!(sigmoid(&x), x_results);

        let x = array![[-1.0, 1.0, 2.0], [-5.0, 5.0, 0.1]];
        let x_results = array![
            [0.26894142, 0.731_058_6, 0.880797],
            [0.006692851, 0.993_307_2, 0.524_979_2]
        ];
        assert_eq!(sigmoid(&x), x_results);
    }

    #[test]
    fn test_softmax() {
        let x = array![[3.2, 1.3, 4.2], [0.3, 2.9, 4.0]];
        let x_results = array![
            [0.25854158, 0.0386697, 0.7027887],
            [0.018211273, 0.24519183, 0.73659694]
        ];
        assert_eq!(softmax(&x), x_results);
    }

    #[test]
    fn test_cross_entropy_error() {
        let x = array![[0.3, 0.2, 0.5], [0.2, 0.1, 0.7]];
        let t = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x_result = 0.98305607;
        assert_eq!(cross_entropy_error(&x, &t), x_result);

        let t = array![[1.0], [2.0]];
        assert_eq!(cross_entropy_error(&x, &t), x_result);
    }

    #[test]
    fn test_im2col() {
        let x = Array4::from_shape_vec(
            (2, 3, 3, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 1.0, 2.0,
                3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
            ],
        )
        .unwrap();
        let result: NNMatrix = array![
            [1., 2., 4., 5., 10., 11., 13., 14., 19., 20., 22., 23.],
            [2., 3., 5., 6., 11., 12., 14., 15., 20., 21., 23., 24.],
            [4., 5., 7., 8., 13., 14., 16., 17., 22., 23., 25., 26.],
            [5., 6., 8., 9., 14., 15., 17., 18., 23., 24., 26., 27.],
            [1., 2., 4., 5., 10., 11., 13., 14., 19., 20., 22., 23.],
            [2., 3., 5., 6., 11., 12., 14., 15., 20., 21., 23., 24.],
            [4., 5., 7., 8., 13., 14., 16., 17., 22., 23., 25., 26.],
            [5., 6., 8., 9., 14., 15., 17., 18., 23., 24., 26., 27.]
        ];
        let result2: NNMatrix = array![
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 10., 0., 0., 0., 19.],
            [0., 0., 1., 2., 0., 0., 10., 11., 0., 0., 19., 20.],
            [0., 0., 2., 3., 0., 0., 11., 12., 0., 0., 20., 21.],
            [0., 0., 3., 0., 0., 0., 12., 0., 0., 0., 21., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 4., 0., 10., 0., 13., 0., 19., 0., 22.],
            [1., 2., 4., 5., 10., 11., 13., 14., 19., 20., 22., 23.],
            [2., 3., 5., 6., 11., 12., 14., 15., 20., 21., 23., 24.],
            [3., 0., 6., 0., 12., 0., 15., 0., 21., 0., 24., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 4., 0., 7., 0., 13., 0., 16., 0., 22., 0., 25.],
            [4., 5., 7., 8., 13., 14., 16., 17., 22., 23., 25., 26.],
            [5., 6., 8., 9., 14., 15., 17., 18., 23., 24., 26., 27.],
            [6., 0., 9., 0., 15., 0., 18., 0., 24., 0., 27., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 7., 0., 0., 0., 16., 0., 0., 0., 25., 0., 0.],
            [7., 8., 0., 0., 16., 17., 0., 0., 25., 26., 0., 0.],
            [8., 9., 0., 0., 17., 18., 0., 0., 26., 27., 0., 0.],
            [9., 0., 0., 0., 18., 0., 0., 0., 27., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 10., 0., 0., 0., 19.],
            [0., 0., 1., 2., 0., 0., 10., 11., 0., 0., 19., 20.],
            [0., 0., 2., 3., 0., 0., 11., 12., 0., 0., 20., 21.],
            [0., 0., 3., 0., 0., 0., 12., 0., 0., 0., 21., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 4., 0., 10., 0., 13., 0., 19., 0., 22.],
            [1., 2., 4., 5., 10., 11., 13., 14., 19., 20., 22., 23.],
            [2., 3., 5., 6., 11., 12., 14., 15., 20., 21., 23., 24.],
            [3., 0., 6., 0., 12., 0., 15., 0., 21., 0., 24., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 4., 0., 7., 0., 13., 0., 16., 0., 22., 0., 25.],
            [4., 5., 7., 8., 13., 14., 16., 17., 22., 23., 25., 26.],
            [5., 6., 8., 9., 14., 15., 17., 18., 23., 24., 26., 27.],
            [6., 0., 9., 0., 15., 0., 18., 0., 24., 0., 27., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 7., 0., 0., 0., 16., 0., 0., 0., 25., 0., 0.],
            [7., 8., 0., 0., 16., 17., 0., 0., 25., 26., 0., 0.],
            [8., 9., 0., 0., 17., 18., 0., 0., 26., 27., 0., 0.],
            [9., 0., 0., 0., 18., 0., 0., 0., 27., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ];

        assert_eq!(im2col(&x, 2, 2, 1, 0), result);
        assert_eq!(im2col(&x, 2, 2, 1, 2), result2);
    }
}
