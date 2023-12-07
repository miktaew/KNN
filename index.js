const {Comparator} = require("./KNN_comparator.js");
const {KNN, FuzzyKNN, distances} = require("./KNN.js");
const fs = require('fs');
const {parse} = require('csv-parse'); 
const { performance } = require('perf_hooks');

// https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset 
const run_sms = (distance, m) => {

    const data = [];

    const temp = {};
    fs.createReadStream("./spam.csv", "utf8")
        .pipe(parse({ delimiter: ",", from_line: 2 }))
        .on("data", (row) => {

            //removing duplicates
            if(temp[row[1]]) {
                return;
            }
            temp[row[1]] = 1;
            //
            const special_characters = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/gi;
            const special_count = (row[1].match(special_characters) || []).length;
            const space_count = (row[1].split(" ") || []).length - 1;
            const body_len = row[1].length;

            //multiplying by 100 to make scales more similar, theoretically gives better results
            const coords = [body_len, 100*special_count/body_len, 100*space_count/body_len];
            const class_value = row[0];
            const text = row[1];
            data.push({coords: coords, class: class_value, text});
        }).on('end', () => {

            const comparator = new Comparator(data);
            const res = comparator.run_compare([3,5,7,9,11,13,15,17], 10, distance, m);
            console.log("KNN", res.KNN.map(x => {return {k: x.k, accuracy: x.overall_accuracy}}));
            console.log("FuzzyKNN", res.FuzzyKNN.map(x => {return {k: x.k, accuracy: x.overall_accuracy}}));
        }
    );
}

// https://www.kaggle.com/datasets/uciml/iris 
const run_iris = (distance, m) => {

    const data = [];

    fs.createReadStream("./Iris.csv", "utf8")
        .pipe(parse({ delimiter: ",", from_line: 2 }))
        .on("data", (row) => {
            data.push({coords: [row[0], row[1], row[2], row[3], row[4]], class: row[5]});
        }).on('end', () => {
            const comparator = new Comparator(data);
            const res = comparator.run_compare([7], 10, distance, m);

            console.log("KNN:", res.KNN[0]);
            console.log("FuzzyKNN:", res.FuzzyKNN[0]);
        }
    );
}

// https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
const run_cancer = (distance, m) => {

    const data = [];

    fs.createReadStream("./cancer.csv", "utf8")
        .pipe(parse({ delimiter: ",", from_line: 2 }))
        .on("data", (row) => {

            const coords = [];
            for(let i = 2; i < row.length; i++) {
                coords.push(row[i]);
            }
            data.push({coords, class: row[1]});
        }).on('end', () => {

            const comparator = new Comparator(data);

            const res = comparator.run_compare([3,5,7,9,11,13,15,17], 10, distance, m);
            //console.log("KNN", res.KNN.map(x => {return {k: x.k, accuracy: x.overall_accuracy}}));
            //console.log("FuzzyKNN", res.FuzzyKNN.map(x => {return {k: x.k, accuracy: x.overall_accuracy}}));
            console.log(res.KNN[0]);
        }
    );
}

//run_sms3(distances.EUCLIDEAN, 2);
run_iris(distances.MANHATTAN, 2);
//run_cancer(distances.EUCLIDEAN, 2);


/*
//another tiny example
const data = [
    {coords: [1,1], class: 1},
    {coords: [4,3], class: 2},
    {coords: [3,0], class: 3},
    {coords: [3,1], class: 3},
    {coords: [2,2], class: 1},
    {coords: [4,2], class: 2},
    {coords: [5,2], class: 3},
    {coords: [1,0], class: 2},
    {coords: [3,3], class: 1},
    {coords: [0,0], class: 1}
];

const knn = new KNN();
knn.loadPoints(data);
console.log(knn.fit({pointsToFit: [{coords: [2,1]}], k: 3, distance_method: distances.EUCLIDEAN}));
const fuzzyKNN = new FuzzyKNN();
fuzzyKNN.loadPoints(data);
console.log(fuzzyKNN.fit({pointsToFit: [{coords: [2,1]}], k: 3, distance_method: distances.EUCLIDEAN, m: 2}));

const comparator = new Comparator(data);
const res = comparator.compare([3,5], 4, "FuzzyKNN", distances.EUCLIDEAN, 3);
console.log(res);
*/