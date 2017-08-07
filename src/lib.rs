extern crate utah;
#[macro_use] // ! macro before crate it applies!!!
extern crate ndarray;

//http://archive.ics.uci.edu/ml/datasets/banknote+authentication
//http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
// TODO remove mut and unwrap and other bad stuff

use ndarray::{Array1,Array2,Axis,S,Si,arr1,arr2};
use std::iter::FromIterator;
use std::sync::Arc;
use std::borrow::Borrow;

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
    let mut result : Option<Split> = None;
    match node
    {
        Split::Res{ index, value, groups } => {
            let left = &groups[0];
            let right = &groups[1];
            println!("left {:?} right {:?}",left.dim(),right.dim());
            if left.dim().0 < 2 || right.dim().0 < 2
            {   
                result = Some(Split::Final{
                    index: index,
                    value: value,
                    left: if left.dim().0 == 0 { to_terminal(right) } else { to_terminal(left) },
                    right: if right.dim().0 == 0 { to_terminal(left) } else { to_terminal(right) }
                });
            }
            else if depth >= max_depth
            {
                result = Some(Split::Final{
                    index: index,
                    value: value,
                    left: to_terminal(left),
                    right: to_terminal(right)
                });
            }
            else {
                let mut node_left : Node = Node::Empty;
                let mut node_right : Node = Node::Empty;
                //process left child
                if left.dim().1  <= min_size {
                    node_left = to_terminal(left);
                }
                else {
                    let node_left_result = get_split(left);
                    node_left = Node::Decision(Arc::new(split(node_left_result, max_depth, min_size, depth + 1).unwrap()));
                }
                //process right child
                if right.dim().1  <= min_size {
                    node_right = to_terminal(right);
                }
                else {
                    let node_right_result = get_split(right);
                    node_right = Node::Decision(Arc::new(split(node_right_result, max_depth, min_size, depth + 1).unwrap()));
                }

                result = Some(Split::Final{
                    index: index,
                    value: value,
                    left: node_left,
                    right: node_right
                });
            }
        }
        _ => {
            println!("already processed");
        }
    }
    result
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

#[cfg(test)]
mod tests {
    use {gini_index,test_split,get_split,build_tree,predict_tree,Split,Node};
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

    /*#[test]
    fn tree_case_bank_note()
    {
        assert_eq!(0,1)
    }*/
    
}