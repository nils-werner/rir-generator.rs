mod rir {
    use std::f64::consts::PI;

    #[derive(Debug, Clone)]
    pub enum Microphone {
        Bidirectional,
        Hypercardioid,
        Cardioid,
        Subcardioid,
        Omnidirectional,
    }

    impl Into<f64> for Microphone {
        fn into(self) -> f64 {
            match self {
                Self::Bidirectional => 0.0,
                Self::Hypercardioid => 0.25,
                Self::Cardioid => 0.5,
                Self::Subcardioid => 0.75,
                Self::Omnidirectional => 1.0,
            }
        }
    }

    impl From<char> for Microphone {
        fn from(x: char) -> Microphone {
            match x {
                'b' => Self::Bidirectional,
                'h' => Self::Hypercardioid,
                'c' => Self::Cardioid,
                's' => Self::Subcardioid,
                _ => Self::Omnidirectional,
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

    impl From<&[f64; 3]> for Position {
        fn from(x: &[f64; 3]) -> Position {
            Position {
                x: x[0],
                y: x[1],
                z: x[2],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Betas {
        pub x: [f64; 2],
        pub y: [f64; 2],
        pub z: [f64; 2],
    }

    impl From<f64> for Betas {
        fn from(x: f64) -> Betas {
            Betas {
                x: [x, x],
                y: [x, x],
                z: [x, x],
            }
        }
    }

    impl From<&[f64; 6]> for Betas {
        fn from(x: &[f64; 6]) -> Betas {
            Betas {
                x: [x[0], x[1]],
                y: [x[2], x[3]],
                z: [x[4], x[5]],
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct Angle {
        pub phi: f64,
        pub theta: f64,
    }

    impl From<f64> for Angle {
        fn from(x: f64) -> Angle {
            Angle { phi: x, theta: x }
        }
    }

    impl From<&[f64; 2]> for Angle {
        fn from(x: &[f64; 2]) -> Angle {
            Angle {
                phi: x[0],
                theta: x[1],
            }
        }
    }

    trait FloatSinc {
        fn sinc(self) -> f64;
    }

    impl FloatSinc for f64 {
        fn sinc(self) -> f64 {
            if self < 1.001 && self > 0.999 {
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
                let rho: f64 = t.clone().into();
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
        receivers: &[Position],
        source: &Position,
        room: &Room,
        beta: &Betas,
        microphone_types: &[Microphone],
        microphone_angles: &[Angle],
        n_samples: usize,
        n_order: i64,
        enable_highpass_filter: bool,
    ) -> Vec<Vec<f64>> {
        // Temporary variables and constants (image-method)
        let fc = 0.5; // The normalized cut-off frequency equals (fs/2) / fs = 0.5
        let tw = (2.0 * (0.004 * fs).round()) as usize; // The width of the low-pass FIR equals 8 ms
        let cts = c / fs;
        let mut imp = vec![vec![0.0; n_samples]; receivers.len()];

        let source = Position {
            x: source.x / cts,
            y: source.y / cts,
            z: source.z / cts,
        };
        let room = Room {
            x: room.x / cts,
            y: room.y / cts,
            z: room.z / cts,
        };

        for (i, ((receiver, angle), mtype)) in receivers
            .iter()
            .zip(microphone_angles.iter())
            .zip(microphone_types.iter())
            .enumerate()
        {
            let receiver = Position {
                x: receiver.x / cts,
                y: receiver.y / cts,
                z: receiver.z / cts,
            };

            let n1 = (n_samples as f64 / (2.0 * room.x)).ceil() as i64;
            let n2 = (n_samples as f64 / (2.0 * room.y)).ceil() as i64;
            let n3 = (n_samples as f64 / (2.0 * room.z)).ceil() as i64;

            // Generate room impulse response
            for mx in -n1..=n1 {
                for my in -n2..=n2 {
                    for mz in -n3..=n3 {
                        let rm = Room {
                            x: 2.0 * mx as f64 * room.x,
                            y: 2.0 * my as f64 * room.y,
                            z: 2.0 * mz as f64 * room.z,
                        };

                        for q in 0..=1 {
                            for j in 0..=1 {
                                for k in 0..=1 {
                                    let rp_plus_rm = Room {
                                        x: (1 - 2 * q) as f64 * source.x - receiver.x + rm.x,
                                        y: (1 - 2 * j) as f64 * source.y - receiver.y + rm.y,
                                        z: (1 - 2 * k) as f64 * source.z - receiver.z + rm.z,
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

                                    if n_order == -1
                                        || (2 * mx - q).abs()
                                            + (2 * my - j).abs()
                                            + (2 * mz - k).abs()
                                            <= n_order
                                    {
                                        let fdist = (dist).floor();
                                        if (fdist as usize) < n_samples {
                                            let gain = sim_microphone(&rp_plus_rm, &angle, &mtype)
                                                * refl[0]
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
                                                    imp[i][start_position + n] += gain * lpi[n];
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
        }

        if enable_highpass_filter {
            // 'Original' high-pass filter as proposed by Allen and Berkley.
            // Temporary variables and constants (high-pass filter)
            let w = 2.0 * PI * 100.0 / fs; // The cut-off frequency equals 100 Hz
            let r1 = -w.exp();
            let b1 = 2.0 * r1 * w.cos();
            let b2 = -r1 * r1;
            let a1 = -(1.0 + r1);

            for (i, _) in receivers.iter().enumerate() {
                let mut y = [0.0; 3];

                for n in 0..n_samples {
                    let x0 = imp[i][n];
                    y[2] = y[1];
                    y[1] = y[0];
                    y[0] = b1 * y[1] + b2 * y[2] + x0;
                    imp[i][n] = y[0] + a1 * y[1] + r1 * y[2];
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
        let imp = compute_rir(
            340.0,
            16000.0,
            &vec![Position::from(&[2.0, 1.5, 2.0])],
            &Position::from(&[2.0, 3.5, 2.0]),
            &Room::from(&[5.0, 4.0, 6.0]),
            &Betas::from(0.4),
            &vec![Microphone::Omnidirectional],
            &vec![Angle::from(0.0)],
            4096,
            -1,
            true,
        );

        assert!(imp
            .iter()
            .map(|inner| inner.iter().any(|&x| x > 0.0))
            .all(|x| x))
    }
}
