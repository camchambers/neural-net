
use rand::Rng;
use std::iter::repeat_with;

pub struct Neuron {
    pub weights: Vec<f64>,
    bias: f64,
}

impl Neuron {
    pub fn new(inputs: usize, bias: f64) -> Neuron {
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = repeat_with(|| 2.0 * rng.gen::<f64>() - 1.0).take(inputs + 1).collect();
        Neuron { weights, bias }
    }

    pub fn run(&self, mut x: Vec<f64>) -> f64 {
        x.push(self.bias);
        let sum: f64 = x.iter().zip(&self.weights).map(|(a, b)| a * b).sum();
        sigmoid(sum)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}