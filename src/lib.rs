extern crate utah;
#[macro_use] // ! macro before crate it applies!!!
extern crate ndarray;

//http://archive.ics.uci.edu/ml/datasets/banknote+authentication
//http://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

use ndarray::{Array1,Array2,Axis,S,Si,arr1,arr2};
use std::iter::FromIterator;

pub fn gini_index(class_value: &Array1<u64>,groups: &[Array2<f64>; 2]) -> f64
{
    let mut gini = 0.0;
    for class_val in class_value.iter()
    {
		//println!("value {}", class_val);
        let class_val_f = *class_val as f64;
        for group in groups.iter().filter(|group| group.len_of(Axis(0)) > 0 )
        {
            let size = group.len_of(Axis(0));
            //println!("size {}", size);
            //println!("group sliced {}", group.slice(s![.., -1..]));
            let proportion = (group.slice(s![.., -1..]).iter().filter(|x| **x == class_val_f).count() as f64) / (size as f64);
            //println!("proportion {}", proportion);
            gini += proportion * (1.0 - proportion);
            //println!("gini {}", gini);
        }
    }
    //println!("final gini {}", gini);
    gini
}

pub fn test_split(index: isize, value: f64, dataset: &Array2<f64>) -> [Array2<f64>; 2]
{
    //slice on index, then generate indexes for value, then select index
    let dataset_index = dataset.slice(&[S,Si(index,Some(index+1),1)]);
    //println!("dataset_index {}", dataset_index);
    let leftidx = Vec::from_iter(dataset_index.iter().enumerate().filter(|&(_,val)| *val < value).map(|(i,_)| i));
    let rightidx = Vec::from_iter(dataset_index.iter().enumerate().filter(|&(_,val)| *val >= value).map(|(i,_)| i));
    let left = dataset.select(Axis(0), &leftidx[..]); //convert vec to slice with range operator (it's not an array!)
    //println!("left {}",left);
    let right = dataset.select(Axis(0), &rightidx[..]);
    //println!("right {}",right);
    [
        left,
        right
    ]
}

#[derive(Debug)]
pub struct Split {
    index : isize,
    value : f64,
    groups : [Array2<f64>; 2]
}

impl Split{
    fn new(index : isize, value : f64, groups : [Array2<f64>; 2]) -> Split { 
        Split{
            index : index,
            value : value,
            groups : groups 
        }
    }
}

impl PartialEq for Split {
    fn eq(&self, other: &Split) -> bool {
        self.index == other.index && self.value == other.value && self.groups == other.groups
    }
}

pub fn get_split(dataset: Array2<f64>) -> Split
{
    let mut class_col =  Vec::from_iter(dataset.slice(s![.., -1..]).iter().map(|x| *x as u64));
    class_col.sort();
    class_col.dedup(); //get distinct values
    let class_values = arr1(&class_col.clone()[..]);
    //println!("class_values {}",class_values);
    let colnum = (dataset.dim().1) - 1;
    //println!("colnum {}",colnum);
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
            //println!("groups {:?}", groups);
            let gini = gini_index(&class_values,&groups);
            //println!("gini {:?}", gini);
            println!("X{} < {} Gini = {}",index+1, val, gini);
            if gini < b_score 
            {
                b_index = index as isize;
                b_value = val;
                b_score = gini;
                b_groups = [groups[0].clone(),groups[1].clone()];
            }
        }
        //println!("index {}",index);
    }
    Split::new(
        b_index,
        b_value,
        b_groups
    )
}

#[cfg(test)]
mod tests {
    use {gini_index,test_split,get_split,Split};
    use ndarray::{arr1,arr2};

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
            Split::new(
                0,
                6.642287351,
                [
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
            ),
            get_split(
                arr2(&[[2.771244718,1.784783929,0.0],
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
    fn tree_build()
    {
        assert_eq!(0,1)
    }

    #[test]
    fn tree_predict()
    {
        assert_eq!(0,1)
    }

    #[test]
    fn tree_case_bank_note()
    {
        assert_eq!(0,1)
    }
    
}
