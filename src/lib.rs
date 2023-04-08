mod rir {
    use std::f64::consts::PI;

    extern crate derive_builder;

    // #[derive(Default, Builder, Debug)]
    // #[builder(setter(into))]
    // struct RIR {
    //     rr: &Vec<Position>,
    //     ss: &Position,
    //     ll: &Position,
    //     beta: &Betas,
    //     microphone_type: &Microphone,
    //     n_order: i64,
    //     microphone_angle: &Angle,
    //     #[builder(default = 340)]
    //     c: f64,
    //     #[builder(default = 16000)]
    //     fs: f64,
    //     #[builder(default = 4096)]
    //     n_samples: usize,
    //     #[builder(default = true)]
    //     is_highpass_filter: bool
    // }

    #[derive(Debug)]
    pub enum Microphone {
        Bidirectional,
        Hypercardioid,
        Cardioid,
        Subcardioid,
        Omnidirectional,
    }

    impl Microphone {
        fn value(&self) -> f64 {
            match *self {
                Self::Bidirectional => 0.0,
                Self::Hypercardioid => 0.25,
                Self::Cardioid => 0.5,
                Self::Subcardioid => 0.75,
                Self::Omnidirectional => 1.0,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Position {
        pub x: f64,
        pub y: f64,
        pub z: f64,
    }
    pub type Room = Position;

    // trait FromArray for Position{
    //     fn from_array(arr: &[f64]) -> Position {
    //         Position{ x: arr[0], y: arr[1], z: arr[2]}
    //     }
    // }

    #[derive(Debug)]
    pub struct Betas {
        pub x: [f64; 2],
        pub y: [f64; 2],
        pub z: [f64; 2],
    }

    #[derive(Debug)]
    pub struct Angle {
        pub phi: f64,
        pub theta: f64,
    }

    trait FloatSinc {
        fn sinc(&self) -> f64;
    }

    impl FloatSinc for f64 {
        fn sinc(&self) -> f64 {
            if *self < 1.001 && *self > 0.999 {
                1.0
            } else {
                self.sin() / self
            }
        }
    }

    fn sim_microphone(p: &Position, a: &Angle, t: &Microphone) -> f64 {
        match t {
            Microphone::Hypercardioid => 1.0,
            _ => {
                let rho = t.value();
                let vartheta = (p.z / (p.x.powi(2) + p.y.powi(2) + p.z.powi(2)).sqrt()).acos();
                let varphi = p.y.atan2(p.x);

                let gain = (PI / 2.0 - a.theta).sin() * (vartheta).sin() * (a.phi - varphi).cos()
                    + (PI / 2.0 - a.theta).cos() * (vartheta).cos();
                rho + (1.0 - rho) * gain
            }
        }
    }

    pub fn compute_rir(
        c: f64,
        fs: f64,
        rr: &Vec<Position>,
        n_samples: usize,
        ss: &Position,
        ll: &Position,
        beta: &Betas,
        microphone_type: &Microphone,
        n_order: i64,
        microphone_angle: &Angle,
        is_highpass_filter: bool,
    ) -> Vec<Vec<f64>> {
        // Temporary variables and constants (image-method)
        let fc = 0.5; // The normalized cut-off frequency equals (fs/2) / fs = 0.5
        let tw = (2.0 * (0.004 * fs).round()) as usize; // The width of the low-pass FIR equals 8 ms
        let cts = c / fs;
        let mut imp = vec![vec![0.0; n_samples]; rr.len()];
        // let mut imp = ndarray::Array::zeros((rr.len(), n_samples));

        let s = Position {
            x: ss.x / cts,
            y: ss.y / cts,
            z: ss.z / cts,
        };
        let l = Room {
            x: ll.x / cts,
            y: ll.y / cts,
            z: ll.z / cts,
        };

        for (idx_microphone, r) in rr.iter().enumerate() {
            let r = Position {
                x: r.x / cts,
                y: r.y / cts,
                z: r.z / cts,
            };

            let n1 = (n_samples as f64 / (2.0 * l.x)).ceil() as i64;
            let n2 = (n_samples as f64 / (2.0 * l.y)).ceil() as i64;
            let n3 = (n_samples as f64 / (2.0 * l.z)).ceil() as i64;

            // Generate room impulse response
            for mx in -n1..=n1 {
                for my in -n2..=n2 {
                    for mz in -n3..=n3 {
                        let rm = Room {
                            x: 2.0 * mx as f64 * l.x,
                            y: 2.0 * my as f64 * l.y,
                            z: 2.0 * mz as f64 * l.z,
                        };

                        for q in 0..=1 {
                            for j in 0..=1 {
                                for k in 0..=1 {
                                    let rp_plus_rm = Room {
                                        x: (1 - 2 * q) as f64 * s.x - r.x + rm.x,
                                        y: (1 - 2 * j) as f64 * s.y - r.y + rm.y,
                                        z: (1 - 2 * k) as f64 * s.z - r.z + rm.z,
                                    };
                                    let refl = [
                                        beta.x[0].powi((mx - q).abs() as i32)
                                            * beta.x[1].powi((mx).abs() as i32),
                                        beta.y[0].powi((my - j).abs() as i32)
                                            * beta.y[1].powi((my).abs() as i32),
                                        beta.z[0].powi((mz - k).abs() as i32)
                                            * beta.z[1].powi((mz).abs() as i32),
                                    ];

                                    let dist = (rp_plus_rm.x.powi(2)
                                        + rp_plus_rm.y.powi(2)
                                        + rp_plus_rm.z.powi(2))
                                    .sqrt();

                                    if (2 * mx - q).abs() + (2 * my - j).abs() + (2 * mz - k).abs()
                                        <= n_order
                                        || n_order == -1
                                    {
                                        let fdist = (dist).floor();
                                        if (fdist as usize) < n_samples {
                                            let gain = sim_microphone(
                                                &rp_plus_rm,
                                                &microphone_angle,
                                                &microphone_type,
                                            ) * refl[0]
                                                * refl[1]
                                                * refl[2]
                                                / (4.0 * PI * dist * cts);

                                            let mut lpi = vec![0.0; tw];
                                            for n in 0..tw {
                                                let t = ((n as f64) - 0.5 * (tw as f64) + 1.0)
                                                    - (dist - fdist);
                                                lpi[n] = 0.5
                                                    * (1.0 + (2.0 * PI * t / (tw as f64)).cos())
                                                    * 2.0
                                                    * fc
                                                    * (2.0 * PI * fc * t).sinc();
                                            }
                                            let start_position =
                                                (fdist - (tw as f64 / 2.0) + 1.0) as usize;
                                            for n in 0..tw {
                                                if start_position + n < n_samples {
                                                    imp[idx_microphone][start_position + n] +=
                                                        gain * lpi[n];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 'Original' high-pass filter as proposed by Allen and Berkley.
            if is_highpass_filter {
                // Temporary variables and constants (high-pass filter)
                let w = 2.0 * PI * 100.0 / fs; // The cut-off frequency equals 100 Hz
                let r1 = -w.exp();
                let b1 = 2.0 * r1 * w.cos();
                let b2 = -r1 * r1;
                let a1 = -(1.0 + r1);
                let mut y = [0.0; 3];

                for idx in 0..n_samples {
                    let x0 = imp[idx_microphone][idx];
                    y[2] = y[1];
                    y[1] = y[0];
                    y[0] = b1 * y[1] + b2 * y[2] + x0;
                    imp[idx_microphone][idx] = y[0] + a1 * y[1] + r1 * y[2];
                }
            }
        }

        imp
    }
}

#[cfg(test)]
mod tests {
    use crate::rir::*;

    #[test]
    fn test_example() {
        let _ = compute_rir(
            340.0,
            16000.0,
            &vec![Position {
                x: 2.0,
                y: 1.5,
                z: 2.0,
            }],
            4096,
            &Position {
                x: 2.0,
                y: 3.5,
                z: 2.0,
            },
            &Room {
                x: 5.0,
                y: 4.0,
                z: 6.0,
            },
            &Betas {
                x: [0.4, 0.4],
                y: [0.4, 0.4],
                z: [0.4, 0.4],
            },
            &Microphone::Omnidirectional,
            -1,
            &Angle {
                phi: 0.0,
                theta: 0.0,
            },
            true,
        );
    }
}
