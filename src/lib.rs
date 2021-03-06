//extern crate utah;
#[macro_use] // ! macro before crate it applies!!!
extern crate ndarray;

//CART
//http://archive.ics.uci.edu/ml/datasets/banknote+authentication
//http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

//SVM
//http://tullo.ch/articles/svm-py/
//https://github.com/cran/e1071/blob/master/src/svm.cpp
//https://en.wikipedia.org/wiki/Sequential_minimal_optimization
//http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf
//https://varcity.ethz.ch/paper/cvpr2016_li_fastsvmplus.pdf
//https://www.quora.com/Why-is-Support-vector-Machine-hard-to-code-from-scratch-where-Logistic-Regression-is-not

//NN
//http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

// TODO 
// * remove mut and handle instead of unwrap and other bad stuff
// * make types for categorical, nominal etc
// * model trait or other abstraction
// * add regression
// * add kernels to svm
// * bagging and boosting
// * quadratic programming optimization?
// * organize libs and crates
// * serialize and deserialize trained models (RON, PMML?),
// * distributed computation?

use ndarray::{Array1,Array2,Axis,S,Si,arr1,arr2};
use std::iter::FromIterator;
use std::sync::Arc;

pub fn gini_index(class_value: &Array1<u64>,groups: &[Array2<f64>; 2]) -> f64
{
    //http://techqa.info/programming/question/33600843/borrow-check-error-with-variable-not-living-long-enough-in-nested-lambda
    class_value
        .iter()
        .map(|class_val| *class_val as f64)
        .flat_map(|class_val|
            groups
                .iter()
                .filter(|group| group.len_of(Axis(0)) > 0 )
                .map(move |group| 
                    {
                        (group
                            .slice(s![.., -1..])
                            .iter()
                            .filter(|x| **x == class_val)
                            .count() as f64) 
                        / 
                        (group.len_of(Axis(0)) as f64)
                    }
                )
                .map(|proportion| proportion * (1.0 - proportion))
        )
        .sum()
}

pub fn test_split(index: isize, value: f64, dataset: &Array2<f64>) -> [Array2<f64>; 2]
{
    //slice on index, then generate indexes for value, then select index
    let dataset_index = dataset.slice(&[S,Si(index,Some(index+1),1)]);
    let leftidx = Vec::from_iter(dataset_index.iter().enumerate().filter(|&(_,val)| *val < value).map(|(i,_)| i));
    let rightidx = Vec::from_iter(dataset_index.iter().enumerate().filter(|&(_,val)| *val >= value).map(|(i,_)| i));
    let left = dataset.select(Axis(0), &leftidx[..]); //convert vec to slice with range operator (it's not an array!)
    let right = dataset.select(Axis(0), &rightidx[..]);
    [
        left,
        right
    ]
}

#[derive(Debug)]
#[derive(PartialEq)]
pub enum Split{
    Res{index : isize, value : f64,groups : [Array2<f64>; 2]},
    Final{index : isize, value : f64,left: Node, right: Node}
}

#[derive(Debug)]
#[derive(PartialEq)]
pub enum Node {
    Terminal(u64),
    Decision(Arc<Split>),
    Empty
}

pub fn get_split(dataset: &Array2<f64>) -> Split
{
    let mut class_col =  Vec::from_iter(dataset.slice(s![.., -1..]).iter().map(|x| *x as u64));
    class_col.sort();
    class_col.dedup(); //get distinct values
    let class_values = arr1(&class_col.clone()[..]);
    let colnum = (dataset.dim().1) - 1;
    let mut b_index = 999;
    let mut b_value = 999.0;
    let mut b_score = 999.0;
    let mut b_groups = [arr2(&[[]]),arr2(&[[]])];
    for index in 0..colnum
    {
        for row in dataset.genrows()
        {
            let val = *row.iter().nth(index).unwrap();
            let groups = test_split(index as isize,val,&dataset);
            let gini = gini_index(&class_values,&groups);
            println!("X{} < {} Gini = {}",index+1, val, gini);
            if gini < b_score 
            {
                b_index = index as isize;
                b_value = val;
                b_score = gini;
                b_groups = [groups[0].clone(),groups[1].clone()];
            }
        }
    }
    Split::Res{
        index : b_index,
        value : b_value,
        groups : b_groups
    }
    //https://www.reddit.com/r/rust/comments/29kia3/no_ord_for_f32/
    /*let mut gini_vec = Vec::from_iter(
        (0..colnum)
            .flat_map(|index|
                dataset
                    .outer_iter()
                    .map(move |row| *row.iter().nth(index).unwrap() )
                    .map(move |val| (val, test_split(index as isize,val,&dataset)))
                    .map(move |(val,groups)| (index as isize, val, groups[0], groups[1] , gini_index(&class_values, &groups)))
            )
    );
    gini_vec.sort_by(|a,b| (a.4).partial_cmp(&b.4).unwrap_or(Equal));
    match gini_vec.first()
    {
        Some(max) => Split::Res{ index : max.0, value : max.1, groups : [max.2,max.3]},
        None => Split::Res { index : 999, value: 999.0, groups : [arr2(&[[]]),arr2(&[[]])]}
    }*/
}

pub fn to_terminal(group : &Array2<f64>) -> Node {
    let mut class_values = Vec::from_iter(group.slice(s![.., -1..]).iter().map(|x| *x as u64));
    class_values.sort();
    match class_values.get((class_values.len() / 2))  //zero based index : +1 -1
    {
        Some(class) => { Node::Terminal(*class) }
        None => { Node::Empty }
    }
}

pub fn split(node : Split, max_depth: u64, min_size: usize, depth: u64 ) -> Option<Split> {
    match node
    {
        Split::Res{ index, value, groups } => {
            let left = &groups[0];
            let right = &groups[1];
            println!("left {:?} right {:?}",left.dim(),right.dim());
            if left.dim().0 < 2 || right.dim().0 < 2
            {   
                Some(Split::Final{
                    index: index,
                    value: value,
                    left: if left.dim().0 == 0 { to_terminal(right) } else { to_terminal(left) },
                    right: if right.dim().0 == 0 { to_terminal(left) } else { to_terminal(right) }
                })
            }
            else if depth >= max_depth
            {
               Some(Split::Final{
                    index: index,
                    value: value,
                    left: to_terminal(left),
                    right: to_terminal(right)
                })
            }
            else {
                Some(Split::Final{
                    index: index,
                    value: value,
                    left: if left.dim().1 <= min_size { to_terminal(left) } else { Node::Decision(Arc::new(split(get_split(left), max_depth, min_size, depth + 1).unwrap())) },
                    right: if right.dim().1 <= min_size { to_terminal(right) } else { Node::Decision(Arc::new(split(get_split(right), max_depth, min_size, depth + 1).unwrap())) }
                })
            }
        }
        _ => {
            None
        }
    }
}

pub fn build_tree(train : &Array2<f64>, max_depth: u64, min_size: usize) -> Option<Split> {
    split(get_split(train),max_depth, min_size, 1)
}

pub fn predict_row_tree(model: &Split, data : &Array1<f64> ) -> Option<u64> {
    match *model
    {
        Split::Final{index, value, ref left, ref right} => {
            match data.iter().nth(index as usize)
            {
                Some(rowval) => {
                    if *rowval < value {
                        match *left {
                            Node::Decision(ref arc_split) => {predict_row_tree(&Arc::try_unwrap(arc_split.clone()).unwrap(), data)},
                            Node::Terminal(class) => { Some(class) }
                            Node::Empty => { None }
                        }
                    } else {
                        match *right {
                            Node::Decision(ref arc_split) => { predict_row_tree(&Arc::try_unwrap(arc_split.clone()).unwrap(), data) },
                            Node::Terminal(class) => { Some(class) }
                            Node::Empty => { None }
                        }
                    }
                }
                None => None
            }
        }
        _ => None
    }
}

pub fn predict_tree(model: &Split, data : &Array2<f64> ) -> Array1<Option<u64>> {
    Array1::from_iter(
        data
            .outer_iter()
            .map( |row| 
                predict_row_tree(
                    &model,
                    &row.to_owned()
                )
            )
    )
}

pub fn svm_cost_pegasos(x: &Array2<f64>,y: &Array1<i8>,lambda: f64, iterations: u64) -> Array1<f64> {
    let mut w = Array1::<f64>::zeros((x.dim().1));
    let mut t = 1.0;
    for _ in 1..iterations {
        for tau in (0..(x.dim().0 - 1)).map(|tau| tau as isize){
            let ytau = *y.get((tau as usize)).unwrap() as f64;
            let xtau = x.slice(&[Si(tau,Some(tau+1),1),S]).subview(Axis(0),0).to_owned();
            if  ytau * xtau.dot(&w) < 1.0 {
                w = (1.0-1.0/t) * w + 1.0/(lambda*t)*ytau*xtau;
            }
            else
            {
                w = (1.0-1.0/t) * w
            }
            t += 1.0;
        }
    }
    w
}

pub fn nn_calculate_loss(x: &Array2<f64>, y: &Array1<f64>, w1: &Array1<f64>, b1 : &Array1<f64>,w2: &Array1<f64>, b2 : &Array1<f64>) -> f64
{
    let z1 : Array1<f64> = x.dot(w1) + b1;
    let a1 : Array1<f64> = Array1::from_iter(z1.iter().map(|z1s| z1s.tanh()));
    let z2 : Array1<f64> = a1 * w2 + b2;
    let exp_scores = Array1::from_iter(z2.iter().map(|z2s| z2s.exp()));
    let exp_score_sum : f64 = exp_scores.iter().sum();
    let probs : Array1<f64> = Array1::from_iter(exp_scores.iter().map(|exp_score| exp_score / exp_score_sum ));
    0.0
}

#[cfg(test)]
mod tests {
    use {gini_index,test_split,get_split,build_tree,predict_tree,svm_cost_pegasos,Split,Node};
    use ndarray::{arr1,arr2,Array1};
    use std::sync::Arc;

    #[test]
    fn tree_gini_index_one() {
        assert_eq!(
            1.0,
            gini_index(
                &arr1(&[0, 1]),
                &[
                    arr2(&[[1.0, 1.0],
                           [1.0, 0.0]]),
                    arr2(&[[1.0, 1.0],
                           [1.0, 0.0]])
                ]
            )
        )
    }

    #[test]
    fn tree_gini_index_zero() {
        assert_eq!(
            0.0, 
            gini_index(
                &arr1(&[0, 1]),
                &[
                    arr2(&[[1.0, 0.0],
                           [1.0, 0.0]]),
                    arr2(&[[1.0, 1.0],
                           [1.0, 1.0]])
                ]
            )
        )
    }

    #[test]
    fn tree_test_split()
    {
        assert_eq!(
            [
                arr2(&[[1.0,0.0,1.0],
                       [2.0,0.0,1.0],
                       [3.0,0.0,1.0],
                       [4.0,0.0,1.0]]),
                arr2(&[[5.0,0.0,1.0],
                       [6.0,0.0,1.0],
                       [7.0,0.0,1.0],
                       [8.0,0.0,1.0]])
            ],
            test_split(0,5.0,
                &arr2(&[[1.0,0.0,1.0],
                    [2.0,0.0,1.0],
                    [3.0,0.0,1.0],
                    [4.0,0.0,1.0],
                    [5.0,0.0,1.0],
                    [6.0,0.0,1.0],
                    [7.0,0.0,1.0],
                    [8.0,0.0,1.0]])
            )
        )
    }

    #[test]
    fn tree_get_split()
    {
        assert_eq!(
            Split::Res{
                index: 0,
                value: 6.642287351,
                groups: [
                    arr2(&[[2.771244718, 1.784783929, 0.0],
                           [1.728571309, 1.169761413, 0.0],
                           [3.678319846, 2.81281357, 0.0],
                           [3.961043357, 2.61995032, 0.0],
                           [2.999208922, 2.209014212, 0.0]]),
                    arr2(&[[7.497545867, 3.162953546, 1.0],
                           [9.00220326, 3.339047188, 1.0],
                           [7.444542326, 0.476683375, 1.0],
                           [10.12493903, 3.234550982, 1.0],
                           [6.642287351, 3.319983761, 1.0]])
                ]
            },
            get_split(
                &arr2(&[[2.771244718,1.784783929,0.0],
	                   [1.728571309,1.169761413,0.0],
	                   [3.678319846,2.81281357,0.0],
	                   [3.961043357,2.61995032,0.0],
	                   [2.999208922,2.209014212,0.0],
	                   [7.497545867,3.162953546,1.0],
	                   [9.00220326,3.339047188,1.0],
	                   [7.444542326,0.476683375,1.0],
	                   [10.12493903,3.234550982,1.0],
	                   [6.642287351,3.319983761,1.0]]))
        )
    }
    
    #[test]
    fn tree_build_depth_one()
    {
        assert_eq!(
            Some(
                Split::Final {
                    index: 0, 
                    value: 6.642287351, 
                    left: Node::Terminal(0), 
                    right: Node::Terminal(1) 
                }
            ),
            build_tree(
                &arr2(&[[2.771244718,1.784783929,0.0],
	                   [1.728571309,1.169761413,0.0],
	                   [3.678319846,2.81281357,0.0],
	                   [3.961043357,2.61995032,0.0],
	                   [2.999208922,2.209014212,0.0],
	                   [7.497545867,3.162953546,1.0],
	                   [9.00220326,3.339047188,1.0],
	                   [7.444542326,0.476683375,1.0],
	                   [10.12493903,3.234550982,1.0],
	                   [6.642287351,3.319983761,1.0]]),
                1,
                1
            )
        )
    }

    #[test]
    fn tree_build_depth_two()
    {
        assert_eq!(
            Some(
                Split::Final{ 
                    index: 0, 
                    value: 6.642287351, 
                    left: Node::Decision(
                        Arc::new(
                            Split::Final{ 
                                index: 0, 
                                value: 2.771244718, 
                                left: Node::Terminal(0), 
                                right: Node::Terminal(0)
                            }
                        )
                    ), 
                    right: Node::Decision(
                        Arc::new(
                            Split::Final{ 
                                index: 0, 
                                value: 7.497545867, 
                                left: Node::Terminal(1), 
                                right: Node::Terminal(1) 
                            }
                        )
                    )
                }
            ),
            build_tree(
                &arr2(&[[2.771244718,1.784783929,0.0],
	                   [1.728571309,1.169761413,0.0],
	                   [3.678319846,2.81281357,0.0],
	                   [3.961043357,2.61995032,0.0],
	                   [2.999208922,2.209014212,0.0],
	                   [7.497545867,3.162953546,1.0],
	                   [9.00220326,3.339047188,1.0],
	                   [7.444542326,0.476683375,1.0],
	                   [10.12493903,3.234550982,1.0],
	                   [6.642287351,3.319983761,1.0]]),
                2,
                1
            )
        )
    }

    #[test]
    fn tree_build_depth_three()
    {
        assert_eq!(
            Some(
                Split::Final { 
                    index: 0, 
                    value: 6.642287351, 
                    left: Node::Decision(
                        Arc::new(
                            Split::Final { 
                                index: 0, 
                                value: 2.771244718, 
                                left: Node::Terminal(0), 
                                right: Node::Terminal(0) 
                            }
                        )
                    ), 
                    right: Node::Decision(
                        Arc::new(
                            Split::Final { 
                                index: 0, 
                                value: 7.497545867, 
                                left: Node::Decision(
                                    Arc::new(
                                        Split::Final { 
                                            index: 0, 
                                            value: 7.444542326, 
                                            left: Node::Terminal(1), 
                                            right: Node::Terminal(1) 
                                        }
                                    )
                                ), 
                                right: Node::Decision(
                                    Arc::new(
                                        Split::Final { 
                                            index: 0, 
                                            value: 7.497545867, 
                                            left: Node::Terminal(1), 
                                            right: Node::Terminal(1) 
                                        }
                                    )
                                ) 
                            }
                        )
                    )
                }
            ),
            build_tree(
                &arr2(&[[2.771244718,1.784783929,0.0],
	                   [1.728571309,1.169761413,0.0],
	                   [3.678319846,2.81281357,0.0],
	                   [3.961043357,2.61995032,0.0],
	                   [2.999208922,2.209014212,0.0],
	                   [7.497545867,3.162953546,1.0],
	                   [9.00220326,3.339047188,1.0],
	                   [7.444542326,0.476683375,1.0],
	                   [10.12493903,3.234550982,1.0],
	                   [6.642287351,3.319983761,1.0]]),
                3,
                1
            )
        )
    }

    #[test]
    fn tree_predict()
    {
        let prediction = predict_tree(
                &Split::Final{ 
                    index : 0, 
                    value : 6.642287351, 
                    left : Node::Terminal(0), 
                    right : Node::Terminal(1) 
                },
                &arr2(&[[2.771244718,1.784783929,0.0],
	                   [1.728571309,1.169761413,0.0],
	                   [3.678319846,2.81281357,0.0],
	                   [3.961043357,2.61995032,0.0],
	                   [2.999208922,2.209014212,0.0],
	                   [7.497545867,3.162953546,1.0],
	                   [9.00220326,3.339047188,1.0],
	                   [7.444542326,0.476683375,1.0],
	                   [10.12493903,3.234550982,1.0],
	                   [6.642287351,3.319983761,1.0]])
        );
        let predictionresult = Array1::from_iter((&[0,0,0,0,0,1,1,1,1,1]).iter().map(|x| Some(*x as u64)));
        assert_eq!(prediction, predictionresult)
    }
    
    #[test]
    fn svm_cost_pegasos_w()
    {
        assert_eq!(
            arr1(&[0.15617619723045267, -0.12323605735802477]),
            svm_cost_pegasos(&arr2(&[[2.771244718,1.784783929],
	                   [1.728571309,1.169761413],
	                   [3.678319846,2.81281357],
	                   [3.961043357,2.61995032],
	                   [2.999208922,2.209014212],
	                   [7.497545867,3.162953546],
	                   [9.00220326,3.339047188],
	                   [7.444542326,0.476683375],
	                   [10.12493903,3.234550982],
	                   [6.642287351,3.319983761]]),&arr1(&[-1,-1,-1,-1,-1,1,1,1,1,1]),3.0,10))
    }    
}