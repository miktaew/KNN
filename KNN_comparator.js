const {KNN, fuzzyKNN, distances, FuzzyKNN} = require("./KNN.js");
const {fs} = require("fs");

/**
 * runs passed data through selected K values
 */
class Comparator{ 

    data = {};
    constructor(data){
        
        //find classes, load rows for each
        //filter into separate arrs in data

        for(let i = 0; i < data.length; i++) {
            if(!this.data[data[i].class]) {
                this.data[data[i].class] = [data[i]];
            } else {
                this.data[data[i].class].push(data[i]);
            }
        } 
    }
    /**
     * 
     * @param {Number} k_values 
     * @param {Number} samples_count 
     * @param {*} distance_method 
     * @param {Number} m 
     * @param {Boolean} count_time 
     */
    run_compare = (k_values, cv_samples_count, distance_method, m) => {

        let startTime = performance.now();
        const res1 = this.compare(k_values, cv_samples_count, "KNN", distance_method);
        let endTime = performance.now();
        res1.time = `${endTime - startTime} ms`;

        startTime = performance.now();
        const res2 = this.compare(k_values, cv_samples_count, "FuzzyKNN", distance_method, m);
        endTime = performance.now();
        res2.time = `${endTime - startTime} ms`;
        
        return {KNN: res1, FuzzyKNN: res2};
    }

    /**
     * @description call run_compare instead!
     * @param {Array} k_values array of specific values 
     * @param {Number} samples_count number of samples to test; it will test each N-th piece of data (omitted from learning) as testing data
     * @param {String} selected_algorithm - KNN or FuzzyKNN
     * @param {distances} distance_method from 'distances' object, either distance.EUCLIDEAN or distances.MANHATTAN
     * @param {Number} m param of fuzzy calculation, only used if selected algorithm is fuzzy
     */
    compare = (k_values, samples_count, selected_algorithm, distance_method, m) => {

        if(!Number.isInteger(samples_count)) {
            throw("Number of samples to run must be an integer!");
        }

        const res = [];

        for(let i = 0; i < k_values.length; i++) {
            const temp_res = [];
            for(let j = 0; j < samples_count; j++) {
                const test_data = [];
                const learning_data = [];

                Object.keys(this.data).forEach(key => {
                    test_data.push(...this.data[key].slice(Math.floor(j*this.data[key].length/samples_count), (j+1)*this.data[key].length/samples_count));
                    learning_data.push(...this.data[key].slice(0, j*this.data[key].length/samples_count).concat(this.data[key].slice(Math.floor((j+1)*this.data[key].length/samples_count), this.data[key].length)));
                });

                if(selected_algorithm === "KNN") {
                    const knn = new KNN();;
                    knn.loadPoints(learning_data);
                    temp_res.push(knn.fit({pointsToFit: test_data, k: k_values[i], distance_method: distance_method}));

                } else if(selected_algorithm === "FuzzyKNN") {
                    const fuzzyKNN = new FuzzyKNN();
                    fuzzyKNN.loadPoints(learning_data);
                    temp_res.push(fuzzyKNN.fit({pointsToFit: test_data, k: k_values[i], distance_method: distance_method, m}));
                }
            }
            
            const acc = this.calculate_accuracy(temp_res);
            res.push({k: k_values[i], accuracy: acc[0], detailed_accuracy: acc[1], confusion_matrix: acc[2]})
        }
    
        return res;
    }

    /**
     * 
     * @param {} test_data 
     * @returns
     * @description don't call it directly 
     */
    calculate_accuracy = (test_data) => {
        let accurate = 0;
        let wrong = 0;
        const temp_classes = {};
        const matrix = {};

        for(let i = 0; i < test_data.length; i++) {
            
            Object.values(test_data[i].points).forEach((point)=>{
                if(!temp_classes[point.test_class]) {
                    temp_classes[point.test_class] = [0,0];
                }
                if(!matrix[point.test_class]) {
                    matrix[point.test_class] = {};
                }
                if(!matrix[point.actual_class]) {
                    matrix[point.actual_class] = {};
                }

                temp_classes[point.test_class][1]++;
                if(point.test_class === point.actual_class) {
                    accurate++;
                    temp_classes[point.test_class][0]++;

                    matrix[point.test_class][point.test_class] = matrix[point.test_class][point.test_class] + 1 || 1;
                } else {
                    wrong++;

                    matrix[point.test_class][point.actual_class] = matrix[point.test_class][point.actual_class] + 1 || 1;
                }
            });
        }

        const classes = {};
        for(const temp in temp_classes) {
            classes[temp] = temp_classes[temp][0]/temp_classes[temp][1];
        }


        //used to sort properties so it looks more clear
        const matrix2 = {};
        for(const sub in matrix) {
            matrix2[sub] = {};
            for(const sub2 in matrix) {
                if(matrix[sub][sub2]) {
                    matrix2[sub][sub2] = matrix[sub][sub2];
                }else {
                    matrix2[sub][sub2] = 0;
                }
            }
        }

        return [accurate/(accurate+wrong), classes, matrix2];
    }
}


module.exports = {Comparator};