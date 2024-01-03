#![feature(map_try_insert)]
//! A crate for working with polynomials easily
/// A module for engineering polynomials.
///
/// It only has two functions because I forgot what I was going to do with it.
/// Please leave suggestions in the Github issues.
pub mod engineer;
/// An implementation of polynomial arithmetic with all four basic operations and the `pow` trait.
///
/// # Example usage
/// ## Create a new polynomial from a vec of numbers.
/// ```
/// use polynomials_pro::polynomials::*;
/// let poly=Polynomial::new_from_num_vec(vec![4,5,6,3]);
/// println!("{}",poly); // 4x^3+5x^2+6x+3
/// ```
/// ## Multiply polynomials
/// ```
/// use polynomials_pro::polynomials::*;
/// let poly1 = Polynomial::new_from_num_vec(vec![1, 5, 6]);
/// let poly2 = Polynomial::new_from_num_vec(vec![5, 6, 9, 6]);
/// assert_eq!(poly1 * poly2,Polynomial::new_from_num_vec(vec![5,31,69,87,84,36]));
/// ```
pub mod polynomials {
    use num::{Num, Signed};
    use std::{
        collections::BTreeMap,
        fmt::Display,
        ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign},
    };

    #[derive(Debug, Clone, PartialEq)]
    pub struct Polynomial<T: Num>(Vec<PolynomialTerm<T>>);
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PolynomialTerm<T: Num> {
        pub power: i64,
        pub coefficient: T,
    }
    impl<T: Num + Copy + PartialOrd> Polynomial<T> {
        fn combine_like_terms(&mut self) {
            let mut term_map = BTreeMap::new();
            for term in self.0.iter() {
                if term.coefficient.is_zero() {
                    continue;
                }
                if let Err(occupied) = term_map.try_insert(term.power, term.coefficient) {
                    *occupied.entry.into_mut() = occupied.value + *occupied.entry.get();
                }
            }
            self.0.clear();
            for pair in term_map {
                self.0.push(PolynomialTerm {
                    power: pair.0,
                    coefficient: pair.1,
                });
            }
            self.0.sort_by(|a, b| b.power.cmp(&a.power));
        }
        /// Get terms of a polynomial.
        /// This is guaranteed to be sorted in decreasing PartialOrder of power as well has having no terms with the same power.
        /// i.e. this function gets the terms of the polynomial in simplest form.
        pub fn get_terms(&mut self) -> Vec<PolynomialTerm<T>> {
            self.combine_like_terms();
            self.0.clone()
        }
    }
    impl<T: Num + Copy + PartialOrd> Polynomial<T> {
        /// Pushes a term to the polynomial and sorts the terms into PartialOrder.
        pub fn push_term(&mut self, term: PolynomialTerm<T>) {
            self.0.push(term);
            self.0.sort_by(|a, b| b.power.cmp(&a.power));
            self.combine_like_terms();
        }
    }
    impl<T: Num> Polynomial<T> {
        pub fn get_degree(&self) -> i64 {
            for term in &self.0 {
                if term.coefficient != T::zero() {
                    return term.power;
                }
            }
            0
        }
    }
    impl<T: Num + Copy + PartialOrd> Polynomial<T> {
        pub fn new_from_term_vec(terms: Vec<PolynomialTerm<T>>) -> Self {
            let mut new = Polynomial(Vec::new());
            for term in terms {
                new.push_term(term);
            }
            new
        }
        /// This function takes a vector of numbers, where each element has a power one less than the element before it, with the first element having the power of the length of the vector.
        pub fn new_from_num_vec(terms: Vec<T>) -> Self {
            let degree = terms.len() - 1;
            let mut polynomial_terms: Vec<PolynomialTerm<T>> = Vec::new();
            for (i, term) in terms.iter().enumerate() {
                polynomial_terms.push(PolynomialTerm {
                    power: (degree - i) as i64,
                    coefficient: *term,
                });
            }
            Self(polynomial_terms)
        }
    }
    impl<T: Num + Copy + PartialOrd + num::pow::Pow<i64, Output = T> + AddAssign> Polynomial<T> {
        pub fn evaluate(&mut self, x: T) -> T {
            let mut answer = T::zero();
            for term in self.get_terms() {
                answer += term.coefficient * x.pow(term.power);
            }
            answer
        }
    }
    impl<T: Num + Signed + Display> Display for Polynomial<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let mut formatted_equation = String::new();
            for (i, term) in self.0.iter().enumerate() {
                if term.coefficient.is_zero() {
                    continue;
                }
                if i != 0 && term.coefficient.is_positive() {
                    formatted_equation += "+"
                }
                if !term.coefficient.is_one() || term.power == 0 {
                    formatted_equation += &term.coefficient.to_string();
                }
                if term.power != 0 {
                    formatted_equation += "x";
                    if term.power != 1 {
                        formatted_equation += &format!("^{}", term.power);
                    }
                }
            }
            write!(f, "{}", formatted_equation)
        }
    }
    impl<T: Num + Copy + PartialOrd> Mul<Polynomial<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: Polynomial<T>) -> Self::Output {
            let mut out = Polynomial::new_from_term_vec(Vec::new());

            for lht in self.0 {
                for rht in rhs.0.iter() {
                    out.push_term(lht * (*rht));
                }
            }
            out
        }
    }
    impl<T> MulAssign<Polynomial<T>> for Polynomial<T>
    where
        T: Num + Copy + PartialOrd,
    {
        fn mul_assign(&mut self, rhs: Self) {
            *self = self.clone() * rhs;
        }
    }

    impl<T: Num> Mul<PolynomialTerm<T>> for PolynomialTerm<T> {
        type Output = PolynomialTerm<T>;

        fn mul(self, rhs: PolynomialTerm<T>) -> Self::Output {
            PolynomialTerm {
                power: self.power + rhs.power,
                coefficient: self.coefficient * rhs.coefficient,
            }
        }
    }
    impl<T: Num + Copy> Mul<PolynomialTerm<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: PolynomialTerm<T>) -> Self::Output {
            let out = self.clone().0.iter().map(|x| *x * rhs).collect();
            Polynomial(out)
        }
    }

    impl<T: Num + Copy> Mul<T> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self * PolynomialTerm {
                coefficient: rhs,
                power: 0,
            }
        }
    }
    impl<T: Num + Copy + PartialOrd> Add for Polynomial<T> {
        type Output = Polynomial<T>;

        fn add(self, rhs: Self) -> Self::Output {
            let mut cloned = self.clone();
            cloned.0.extend(rhs.0);
            cloned.combine_like_terms();
            cloned
        }
    }

    impl<T: Num + Copy + PartialOrd> AddAssign for Polynomial<T> {
        fn add_assign(&mut self, rhs: Self) {
            *self = self.clone() + rhs
        }
    }
    impl<T: Num + Copy + Signed + PartialOrd> Sub for Polynomial<T> {
        type Output = Polynomial<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            // Negate all terms in rhs then add to polynomial
            let mut cloned = rhs.clone();
            for term in cloned.0.iter_mut() {
                term.coefficient = -term.coefficient
            }
            self + cloned
        }
    }
    impl<T: Num + Copy + PartialOrd + Signed> SubAssign for Polynomial<T> {
        fn sub_assign(&mut self, rhs: Self) {
            *self = self.clone() - rhs
        }
    }
    pub struct DivOutput<T: Num> {
        pub remainder: Polynomial<T>,
        pub output: Polynomial<T>,
    }
    impl<T: Num> Div for PolynomialTerm<T> {
        type Output = PolynomialTerm<T>;

        fn div(self, rhs: Self) -> Self::Output {
            Self {
                power: self.power - rhs.power,
                coefficient: self.coefficient / rhs.coefficient,
            }
        }
    }
    impl<T: Num + Copy> Div<PolynomialTerm<T>> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn div(self, rhs: PolynomialTerm<T>) -> Self::Output {
            let mut out = self;
            for term in out.0.iter_mut() {
                *term = *term / rhs;
            }
            out
        }
    }
    impl<T: Num + Copy + PartialOrd> num::pow::Pow<i32> for Polynomial<T> {
        type Output = Polynomial<T>;

        fn pow(self, rhs: i32) -> Self::Output {
            let mut poly = Polynomial::new_from_num_vec(vec![T::one()]);
            for _ in 0..rhs {
                poly *= self.clone();
            }
            todo!()
        }
    }
    impl<T: Num + Display + Copy + PartialOrd + Signed> Display for PolynomialTerm<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let poly = Polynomial::new_from_term_vec(vec![*self]);

            write!(f, "{}", poly)
        }
    }
    impl<T: Num + Copy + PartialOrd + Signed> Div<Polynomial<T>> for Polynomial<T> {
        type Output = DivOutput<T>;
        // Implementation of polynomial long division.
        fn div(self, rhs: Self) -> Self::Output {
            let binding = rhs.clone();
            let first_divisor_term = binding.0.get(0).unwrap();
            let mut answer = Polynomial::new_from_term_vec(Vec::new());
            let mut remainder = self;
            while remainder.get_degree() >= rhs.get_degree() {
                let factor = *remainder.0.get(0).unwrap() / *first_divisor_term;
                remainder = remainder.clone() - (rhs.clone() * factor);
                answer.push_term(factor);
            }
            remainder.combine_like_terms();
            answer.combine_like_terms();
            DivOutput {
                remainder,
                output: answer,
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::polynomials::*;
    #[test]
    fn polynomial_div() {
        let poly1 = Polynomial::new_from_num_vec(vec![1, 5, 6]);
        let poly2 = Polynomial::new_from_num_vec(vec![5, 6, 9, 6]);
        let combined = poly1.clone() * poly2.clone();
        let divided = combined / poly1;
        assert_eq!(divided.output, poly2);
    }
    #[test]
    fn polynomial_div_with_remainder() {
        let dividend = Polynomial::new_from_num_vec(vec![1., -12., 38., -17.]);
        let divisor = Polynomial::new_from_num_vec(vec![3., 18., 14.]);
        assert_eq!(
            (dividend / divisor).remainder,
            Polynomial::new_from_num_vec(vec![424. / 3., 67.])
        )
    }
    #[test]
    fn polynomial_mul() {
        let poly1 = Polynomial::new_from_num_vec(vec![1, 5, 6]);
        let poly2 = Polynomial::new_from_num_vec(vec![5, 6, 9, 6]);
        assert_eq!(
            poly1 * poly2,
            Polynomial::new_from_num_vec(vec![5, 31, 69, 87, 84, 36])
        );
    }
}
