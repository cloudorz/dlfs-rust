use crate::functions::*;
use crate::types::*;
use ndarray::{Array2, Axis};

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
