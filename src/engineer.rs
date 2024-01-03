use std::ops::Neg;

use num::{Num, traits::Pow};

use crate::polynomials::{Polynomial, PolynomialTerm};
pub fn create_polynomial_with_zeros<T: Num + Neg<Output = T> + Ord + Copy>(
    zeros: Vec<T>,
) -> Polynomial<T> {
    // Create a neutral polynomial.
    let mut poly: Polynomial<T> = Polynomial::new_from_num_vec(vec![T::one()]);
    for zero in zeros {
        let temp_poly: Polynomial<T> = Polynomial::new_with_term_vec(vec![
            PolynomialTerm {
                coefficient: -zero,
                power: 0,
            },
            PolynomialTerm {
                coefficient: T::one(),
                power: 1,
            },
        ]);
        poly *= temp_poly
    }
    poly
}

pub fn create_quadratic_with_vertex<T: Num+Neg<Output=T>+Copy+Ord>(vertex:(T,T),leading_coefficient: T)-> Polynomial<T>{
    (Polynomial::new_from_num_vec(vec![leading_coefficient])*Polynomial::new_from_num_vec(vec![T::one(),-vertex.0]).pow(2))+Polynomial::new_from_num_vec(vec![vertex.1])
}