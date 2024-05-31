use crate::functions::*;
use crate::types::*;
use ndarray::{Array, Array1, Array2, Array4, Axis, Dimension};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub type Parameter<'a> = (&'a mut NNMatrix, &'a NNMatrix);

pub trait Layer {
    fn forward(&mut self, x: &NNMatrix) -> NNMatrix;
    fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix;
    fn parameters(&mut self) -> Vec<Parameter>;
}

#[derive(Debug)]
pub struct Relu {
    #[allow(dead_code)]
    mask: Option<Array2<bool>>,
}

impl Relu {
    pub fn new() -> Self {
        Self { mask: None }
    }
}

impl Layer for Relu {
    fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        self.mask = Some(x.mapv(|x| x < 0.0));

        let mut clone_x = x.clone();
        clone_x.zip_mut_with(self.mask.as_ref().unwrap(), |x_value, bool_value| {
            if *bool_value {
                *x_value = 0.0
            };
        });
        clone_x
    }

    fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let mut clone_d_out = d_out.clone();
        clone_d_out.zip_mut_with(self.mask.as_ref().unwrap(), |x_value, bool_value| {
            if *bool_value {
                *x_value = 0.0
            };
        });
        clone_d_out
    }

    fn parameters(&mut self) -> Vec<Parameter> {
        vec![]
    }
}

#[derive(Debug)]
pub struct Sigmoid {
    #[allow(dead_code)]
    out: Option<NNMatrix>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { out: None }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        let out = sigmoid(x);
        self.out = Some(out.clone());
        out
    }

    fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let out = self.out.as_ref().unwrap();

        d_out * (1.0 - out) * out
    }

    fn parameters(&mut self) -> Vec<Parameter> {
        vec![]
    }
}

#[derive(Debug)]
pub struct Affine {
    bias: NNMatrix,
    d_bias: Option<NNMatrix>,
    weight: NNMatrix,
    d_weight: Option<NNMatrix>,
    x: Option<NNMatrix>,
}

impl Affine {
    pub fn new(weight: NNMatrix, bias: NNMatrix) -> Self {
        Self {
            bias,
            d_bias: None,
            weight,
            d_weight: None,
            x: None,
        }
    }
}

impl Affine {}

impl Layer for Affine {
    fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        self.x = Some(x.clone());

        x.dot(&self.weight) + &self.bias
    }

    fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let d_x = d_out.dot(&self.weight.t());
        self.d_weight = Some(self.x.as_ref().unwrap().t().dot(d_out));
        self.d_bias = Some(
            d_out
                .sum_axis(Axis(0))
                .into_shape((1, d_out.shape()[1]))
                .unwrap(),
        );

        d_x
    }

    fn parameters(&mut self) -> Vec<Parameter> {
        vec![
            (&mut self.weight, self.d_weight.as_ref().unwrap()),
            (&mut self.bias, self.d_bias.as_ref().unwrap()),
        ]
    }
}

#[derive(Debug)]
pub struct SoftmaxWithLoss {
    loss: Option<NNFloat>,
    y: Option<NNMatrix>,
    t: Option<NNMatrix>,
}

impl SoftmaxWithLoss {
    pub fn new() -> Self {
        Self {
            loss: None,
            y: None,
            t: None,
        }
    }
}

impl SoftmaxWithLoss {
    pub fn forward(&mut self, x: &NNMatrix, t: &NNMatrix) -> NNFloat {
        self.t = Some(t.clone());
        self.y = Some(softmax(x));
        self.loss = Some(cross_entropy_error(
            self.y.as_ref().unwrap(),
            self.t.as_ref().unwrap(),
        ));

        self.loss.unwrap()
    }

    pub fn backward(&self) -> NNMatrix {
        let ref_t = self.t.as_ref().unwrap();
        let ref_y = self.y.as_ref().unwrap();
        let batch_size = ref_t.shape()[0] as NNFloat;

        if ref_y.len() == ref_t.len() {
            (ref_y - ref_t) / batch_size
        } else {
            let mut dx = ref_y.clone();
            for (i, arr) in ref_t.axis_iter(Axis(0)).enumerate() {
                for value in arr.iter() {
                    dx[[i, *value as usize]] -= 1.0;
                }
            }

            dx / batch_size
        }
    }
}

pub struct Sequence {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequence {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
}

impl Layer for Sequence {
    fn forward(&mut self, x: &NNMatrix) -> NNMatrix {
        let mut x = x.to_owned();
        for layer_box in self.layers.iter_mut() {
            x = layer_box.forward(&x);
        }

        x
    }

    fn backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let mut d_out = d_out.to_owned();
        for layer_box in self.layers.iter_mut().rev() {
            d_out = layer_box.backward(&d_out);
        }

        d_out
    }

    fn parameters(&mut self) -> Vec<Parameter> {
        self.layers
            .iter_mut()
            .flat_map(|x| x.parameters())
            .collect()
    }
}

// http://arxiv.org/abs/1502.03167
pub struct BatchNormalization {
    gamma: NNMatrix,
    beta: NNMatrix,
    momentum: NNFloat,

    running_mean: Option<NNMatrix>,
    running_var: Option<NNMatrix>,

    batch_size: Option<usize>,
    xc: Option<NNMatrix>,
    xn: Option<NNMatrix>,
    std: Option<NNMatrix>,
    d_gamma: Option<NNMatrix>,
    d_beta: Option<NNMatrix>,
}

impl BatchNormalization {
    pub fn new(
        gamma: NNMatrix,
        beta: NNMatrix,
        momentum: NNFloat,
        running_mean: Option<NNMatrix>,
        running_var: Option<NNMatrix>,
    ) -> Self {
        Self {
            gamma,
            beta,
            momentum,
            running_mean,
            running_var,
            batch_size: None,
            xc: None,
            xn: None,
            std: None,
            d_gamma: None,
            d_beta: None,
        }
    }

    pub fn default(gamma: NNMatrix, beta: NNMatrix) -> Self {
        Self::new(gamma, beta, 0.9, None, None)
    }
}

impl BatchNormalization {
    fn forward<T: Dimension>(
        &mut self,
        x: Array<NNFloat, T>,
        train_flag: bool,
    ) -> Array<NNFloat, T> {
        let input_shape = x.raw_dim();
        let mut shape = (0, 0);
        let shape_vec = x.shape().to_vec();
        if x.shape().len() == 2 {
            shape = (shape_vec[0], shape_vec[1]);
        } else {
            shape = (shape_vec[0], shape_vec[1] * shape_vec[2] * shape_vec[3]);
        }
        let x = x.into_shape(shape).unwrap();

        let out = self._forward(&x, train_flag);

        out.into_shape(input_shape).unwrap()
    }

    fn _forward(&mut self, x: &NNMatrix, train_flag: bool) -> NNMatrix {
        if self.running_mean.is_none() {
            let count = x.shape()[1];
            self.running_mean = Some(NNMatrix::zeros((1, count)));
            self.running_var = Some(NNMatrix::zeros((1, count)));
        }

        if train_flag {
            let mu = x.mean_axis(Axis(0)).unwrap();
            let xc = x - &mu;
            let var = xc.mapv(|v| v.exp2()).mean_axis(Axis(0)).unwrap();
            let std = (&var + 10e-7).mapv(|v| v.sqrt());
            let xn = &xc / &std;

            self.batch_size = Some(x.shape()[0]);
            self.xc = Some(xc);
            self.xn = Some(xn);
            let count = std.len();
            self.std = Some(std.into_shape((1, count)).unwrap());

            self.running_mean = Some(
                self.momentum * self.running_mean.as_ref().unwrap() + (1.0 - self.momentum) * &mu,
            );
            self.running_var = Some(
                self.momentum * self.running_var.as_ref().unwrap() + (1.0 - self.momentum) * &var,
            );

            &self.gamma * self.xn.as_ref().unwrap() + &self.beta
        } else {
            let xc = x - self.running_mean.as_ref().unwrap();
            let xn = xc / (self.running_var.as_ref().unwrap() + 10e-7).mapv(|v| v.sqrt());

            &self.gamma * xn + &self.beta
        }
    }

    pub fn backward<T: Dimension>(&mut self, d_out: Array<NNFloat, T>) -> Array<NNFloat, T> {
        let input_shape = d_out.raw_dim();
        let mut shape = (0, 0);
        let shape_vec = d_out.shape().to_vec();
        if d_out.shape().len() != 2 {
            shape = (shape_vec[0], shape_vec[1] * shape_vec[2] * shape_vec[3]);
        } else {
            shape = (shape_vec[0], shape_vec[1]);
        }
        let x = d_out.into_shape(shape).unwrap();

        let out = self._backward(&x);

        out.into_shape(input_shape).unwrap()
    }

    fn _backward(&mut self, d_out: &NNMatrix) -> NNMatrix {
        let sh = (1, d_out.shape()[1]);
        let d_beta = d_out.sum_axis(Axis(0)).into_shape(sh);
        let d_gamma = (self.xn.as_ref().unwrap() * d_out)
            .sum_axis(Axis(0))
            .into_shape(sh);

        let d_xn = &self.gamma * d_out;
        let ref_std = self.std.as_ref().unwrap();
        let mut d_xc = &d_xn / ref_std;
        let ref_xc = self.xc.as_ref().unwrap();
        let d_std = ((&d_xn * ref_xc) / (ref_std * ref_std)).sum_axis(Axis(0));
        let d_var = 0.5 * d_std / ref_std;
        let batch_size = self.batch_size.unwrap() as NNFloat;
        d_xc = d_xc + (2.0 / batch_size) * ref_xc * d_var;
        let d_mu = d_xc.sum_axis(Axis(0));
        let d_x = d_xc - d_mu / batch_size;

        self.d_gamma = d_gamma.ok();
        self.d_beta = d_beta.ok();

        d_x
    }
}

// http://arxiv.org/abs/1207.0580
pub struct Dropout {
    dropout_ratio: NNFloat,
    mask: Option<Array2<bool>>,
}

impl Dropout {
    pub fn new(dropout_ratio: NNFloat) -> Self {
        Self {
            dropout_ratio,
            mask: None,
        }
    }

    pub fn default() -> Self {
        Self::new(0.5)
    }
}

impl Dropout {
    pub fn forward(&mut self, mut x: NNMatrix, train_flag: bool) -> NNMatrix {
        if train_flag {
            let mask = NNMatrix::random(x.raw_dim(), Uniform::new(-1.0, 1.0))
                .mapv(|value| value > self.dropout_ratio);
            x.zip_mut_with(&mask, |value, mask_value| {
                if !mask_value {
                    *value = 0.0;
                }
            });
            self.mask = Some(mask);

            x
        } else {
            x * (1.0 - self.dropout_ratio)
        }
    }

    pub fn backward(&self, mut d_out: NNMatrix) -> NNMatrix {
        d_out.zip_mut_with(self.mask.as_ref().unwrap(), |value, mask_value| {
            if !mask_value {
                *value = 0.0;
            }
        });

        d_out
    }
}

pub struct Convolution {
    weight: Array4<NNFloat>,
    bias: Array1<NNFloat>,
    stride: usize,
    pad: usize,

    x_shape: Option<[usize; 4]>,
    col: Option<NNMatrix>,
    col_w: Option<NNMatrix>,

    d_w: Option<Array4<NNFloat>>,
    d_b: Option<Array1<NNFloat>>,
}

impl Convolution {
    pub fn new(weight: Array4<NNFloat>, bias: Array1<NNFloat>, stride: usize, pad: usize) -> Self {
        Self {
            weight,
            bias,
            stride,
            pad,
            x_shape: None,
            col: None,
            col_w: None,
            d_w: None,
            d_b: None,
        }
    }

    pub fn default(weight: Array4<NNFloat>, bias: Array1<NNFloat>) -> Self {
        Self::new(weight, bias, 1, 0)
    }
}

impl Convolution {
    pub fn forward(&mut self, x: Array4<NNFloat>) -> Array4<NNFloat> {
        let weight_shape = self.weight.shape();
        let filter_number = weight_shape[0];
        let filter_height = weight_shape[2];
        let filter_width = weight_shape[3];
        let x_shape = x.shape();
        let input_number = x_shape[0];
        let channel_count = x_shape[1];
        let input_height = x_shape[2];
        let input_width = x_shape[3];

        let out_height = (input_height + 2 * self.pad - filter_height) / self.stride + 1;
        let out_width = (input_width + 2 * self.pad - filter_width) / self.stride + 1;

        let col = im2col(&x, filter_height, filter_width, self.stride, self.pad);
        let col_w = self
            .weight
            .clone()
            .into_shape((filter_number, channel_count * filter_height * filter_width))
            .unwrap();
        let out = col.dot(&col_w) + &self.bias;
        let mut out = out
            .into_shape((input_number, out_height, out_width, filter_number))
            .unwrap();
        // (0, 1, 2, 3) -> (0, 3, 1, 2)
        out.swap_axes(2, 3);
        out.swap_axes(1, 2);

        self.x_shape = Some([input_number, channel_count, input_height, input_width]);
        self.col = Some(col);
        self.col_w = Some(col_w);

        out
    }

    pub fn backward(&mut self, d_out: Array4<NNFloat>) -> Array4<NNFloat> {
        let weight_shape = self.weight.shape();
        let filter_number = weight_shape[0];
        let channel_count = weight_shape[1];
        let filter_height = weight_shape[2];
        let filter_width = weight_shape[3];
        let mut d_out = d_out;
        // (0, 1, 2, 3) -> (0, 2, 3, 1)
        d_out.swap_axes(1, 3);
        d_out.swap_axes(1, 2);
        let d1 = d_out.shape()[0];
        let d2 = d_out.shape()[1];
        let d3 = d_out.shape()[2];
        let d4 = d_out.shape()[3];
        let d_out = d_out.into_shape((d1 * d2 * d3, d4)).unwrap();

        self.d_b = Some(d_out.sum_axis(Axis(0)));
        let mut d_w = self.col.as_ref().unwrap().t().dot(&d_out);
        d_w.swap_axes(0, 1);
        self.d_w = Some(
            d_w.into_shape((filter_number, channel_count, filter_height, filter_width))
                .unwrap(),
        );

        let d_col = d_out.dot(&self.col_w.as_ref().unwrap().t());

        col2im(
            &d_col,
            self.x_shape.as_ref().unwrap(),
            filter_height,
            filter_width,
            self.stride,
            self.pad,
        )
    }
}

pub struct Pooling {
    pool_h: usize,
    pool_w: usize,
    stride: usize,
    pad: usize,

    x_shape: Option<[usize; 4]>,
    arg_max: Option<Vec<usize>>,
}

impl Pooling {
    pub fn new(pool_h: usize, pool_w: usize, stride: usize, pad: usize) -> Self {
        Self {
            pool_h,
            pool_w,
            stride,
            pad,

            x_shape: None,
            arg_max: None,
        }
    }
}

impl Pooling {
    pub fn forward(&mut self, x: Array4<NNFloat>) -> Array4<NNFloat> {
        let x_shape = x.shape();
        let input_number = x_shape[0];
        let channel_count = x_shape[1];
        let input_height = x_shape[2];
        let input_width = x_shape[3];
        // TODO: no padding?
        let out_height = (input_height + 2 * self.pad - self.pool_h) / self.stride + 1;
        let out_width = (input_width + 2 * self.pad - self.pool_w) / self.stride + 1;

        let col = im2col(&x, self.pool_h, self.pool_w, self.stride, self.pad);
        let col = col
            .into_shape((
                input_number * out_height * out_width * channel_count,
                self.pool_w * self.pool_h,
            ))
            .unwrap();

        let mut arg_max_vec: Vec<usize> = vec![];
        let mut out_vec: Vec<NNFloat> = vec![];
        for row in col.axis_iter(Axis(0)) {
            let value = *row
                .iter()
                .max_by(|x1, x2| x1.partial_cmp(x2).unwrap())
                .unwrap();
            let pos = row.iter().position(|item| *item == value).unwrap();
            arg_max_vec.push(pos);
            out_vec.push(value);
        }
        let mut out = Array4::from_shape_vec(
            (input_number, out_height, out_width, channel_count),
            out_vec,
        )
        .unwrap();
        // (0, 1, 2, 3) -> (0, 3, 1, 2)
        out.swap_axes(2, 3);
        out.swap_axes(1, 2);

        self.x_shape = Some([input_number, channel_count, input_height, input_width]);
        self.arg_max = Some(arg_max_vec);

        out
    }

    pub fn backward(&self, d_out: Array4<NNFloat>) -> Array4<NNFloat> {
        let mut d_out = d_out;
        let mut d_shape = d_out.shape().to_vec();
        // (0, 1, 2, 3) -> (0, 2, 3, 1)
        d_out.swap_axes(1, 3);
        d_out.swap_axes(1, 2);

        let pool_size = self.pool_h * self.pool_w;
        let mut d_max = NNMatrix::zeros((d_out.len(), pool_size));
        let arg_max = self.arg_max.as_ref().unwrap();
        let d_out_vec = d_out.into_raw_vec(); // TODO: right order?
        for i in 0..arg_max.len() {
            d_max[[i, arg_max[i]]] = d_out_vec[i];
        }
        d_shape.push(pool_size);
        let d_max = d_max.into_shape(d_shape.push(pool_size)).unwrap();
        let d_col = d_max
            .into_shape((
                d_shape[0] * d_shape[1] * d_shape[2],
                d_shape[3] * d_shape[4],
            ))
            .unwrap();

        col2im(
            &d_col,
            self.x_shape.as_ref().unwrap(),
            self.pool_h,
            self.pool_w,
            self.stride,
            self.pad,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_relu_forward() {
        let mut relu_layer = Relu::new();
        let x_vec = array![[-1.0, 0.0, 1.0], [100.0, -10.0, 0.0],];
        let result_vec = array![[0.0, 0.0, 1.0], [100.0, 0.0, 0.0],];
        assert_eq!(relu_layer.forward(&x_vec), result_vec);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu_layer = Relu {
            mask: Some(array![[false, true, false], [true, false, true]]),
        };
        let d_out_vec = array![[0.3, -0.5, -2.8], [0.1, 0.4, -1.2]];
        let result_vec = array![[0.3, 0.0, -2.8], [0.0, 0.4, 0.0]];
        assert_eq!(relu_layer.backward(&d_out_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_forward() {
        // a bit of silly case?
        let mut sigmoid_layer = Sigmoid::new();
        let x_vec = array![[18.0, 0.0, -100.0], [-100.0, 50.0, 0.0],];
        let result_vec = array![[1.0, 0.5, 0.0], [0.0, 1.0, 0.5],];
        assert_eq!(sigmoid_layer.forward(&x_vec), result_vec);
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut sigmoid_layer = Sigmoid::new();
        let x_vec = array![[18.0, 0.0, -100.0], [-100.0, 50.0, 0.0],];
        let d_out = array![[1.0, 0.0, -1.0], [-0.5, 1.5, 0.1]];
        let result_vec = array![[1.522_997_9e-8, 0.0, 0.0], [0.0, 0.0, 2.5e-02]];
        sigmoid_layer.forward(&x_vec);

        assert_matrix_eq(&sigmoid_layer.backward(&d_out), &result_vec);
    }

    #[test]
    fn test_softmax_with_loss() {
        let mut softmax_with_loss_layer = SoftmaxWithLoss::new();
        let x_vec = array![[8.1, 2.5, 7.8, 0.5, 2.5], [1.5, 0.3, 0.4, 0.5, 0.6]];
        let t_vec = array![[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0]];
        let t_vec_2 = array![[0.0], [2.0]];
        let result_forward = 1.268945;
        let result_backword = array![
            [
                -2.140_756_7e-1,
                1.057_309_2e-3,
                2.118_179_5e-1,
                1.430_912_4e-4,
                1.057_309_2e-3
            ],
            [
                2.075_968_5e-1,
                6.252_697e-2,
                -4.308_97e-1,
                7.637_061e-2,
                8.440_258e-2
            ]
        ];

        assert_eq_on_epsilon(
            softmax_with_loss_layer.forward(&x_vec, &t_vec),
            result_forward,
        );
        assert_matrix_eq(&softmax_with_loss_layer.backward(), &result_backword);

        assert_eq_on_epsilon(
            softmax_with_loss_layer.forward(&x_vec, &t_vec_2),
            result_forward,
        );
        assert_matrix_eq(&softmax_with_loss_layer.backward(), &result_backword);
    }

    #[test]
    fn test_affine() {
        let weights = array![
            [-0.00208241, 0.00527261, 0.00348774],
            [0.00909407, 0.01500044, 0.00863171],
            [-0.00988222, 0.00255272, -0.00220252],
            [0.01846247, 0.00793499, 0.01425399],
            [-0.00180464, -0.001775, 0.00317565]
        ];
        let bias = array![[-0.05998749, 0.08272106, -0.39614827]];
        let x = array![
            [
                -0.37080965,
                0.726_695_4,
                0.4991388,
                1.926_014_4,
                1.690_234_9
            ],
            [
                -0.351_198_9,
                0.20009917,
                -1.591_653_6,
                -1.131_140_5,
                -0.16842349
            ]
        ];
        let mut affine = Affine::new(weights, bias);
        let d_out = array![
            [1.263_253_5, -0.77904165, 0.33553704],
            [-0.31014598, 0.674_458_9, -1.507_464_3]
        ];
        let result_forward = array![
            [-0.02503056, 0.10522357, -0.35944732],
            [-0.06228708, 0.07113122, -0.40879843]
        ];
        let result_backward = array![
            [-0.00556793, 0.00269841, -0.01521145, 0.02192384, 0.00016863],
            [
                -0.00105563,
                -0.0057153,
                0.00810686,
                -0.02186162,
                -0.00542464
            ]
        ];

        assert_matrix_eq(&affine.forward(&x), &result_forward);
        assert_matrix_eq(&affine.backward(&d_out), &result_backward);
    }

    fn assert_matrix_eq(x: &NNMatrix, y: &NNMatrix) {
        assert!((x - y)
            .iter()
            .all(|value| { value.abs() < NNFloat::EPSILON }));
    }

    fn assert_eq_on_epsilon(x: NNFloat, y: NNFloat) {
        assert!((x - y).abs() < NNFloat::EPSILON);
    }
}
