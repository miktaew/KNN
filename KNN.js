const distances = {
    EUCLIDEAN: "euclidean",
    MANHATTAN: "manhattan"
};

class KNN {

    points = [];

    /**
     * 
     * @param {Array} listOfPoints list of points for learning, assumes no labels; 
     * Operates on original array, but that shouldn't really be a problem
     */
    loadPoints = (listOfPoints)=>{
        if(!Array.isArray(listOfPoints)) {
            throw("Please pass a list of points");
        }
        this.points = listOfPoints;
    }

    /**
     * 
     * @param {*} pointsToFit - object in form {coords, class}, where coords is a list
     * @param {Number} k - number of neighbors to check
     * @param {String} distance_method - either distances.EUCLIDEAN or distances.MANHATTAN
     * @returns 
     */
    fit = ({pointsToFit, k, distance_method})=>{

        const results = {k: k, points: {}};
        for(let i = 0; i < pointsToFit.length; i++) {
            this.calc_distances(pointsToFit[i], distance_method);
            const nearest = this.get_k_nearest(k);
            const classes = {};
            let max = {test_class: null, count: 0, actual_class: pointsToFit[i].class};

            for(let j = 0; j < k; j++) {
                
                if(classes[nearest[j].class]) {
                    classes[nearest[j].class]++;
                } else {
                    classes[nearest[j].class] = 1;
                }
                if(classes[nearest[j].class] > max.count) {
                    max.count = classes[nearest[j].class];
                    max.test_class = nearest[j].class;
                }
            }

            results.points[pointsToFit[i].coords] = max;   
        }

        return results;
    }

    get_k_nearest = (k) => {
        return this.points.slice(0,k);
    }

    calc_distances = (point, distance_method="euclidean")=>{
        if(distance_method !== "euclidean" && distance_method !== "manhattan") {
            throw("Currently supported distance_methods: 'euclidean' and 'manhattan'");
        }
        for(let i = 0; i < this.points.length; i++) {
            if(distance_method === "euclidean") {
                this.points[i]["distance"] = this.euclidean(point, this.points[i]);
            }
            else if(distance_method === "manhattan") {
                this.points[i]["distance"] = this.manhattan(point, this.points[i]);
            }
        }
        this.points = this.points.sort((a,b) => a.distance - b.distance);
    }

    euclidean = (point_a, point_b)=>{
        if(point_a.coords.length !== point_b.coords.length){
            throw("Error, points have different number of coordinate points!");
        }

        let result = 0;
        for(let i = 0; i < point_a.coords.length; i++) {
            result += (point_a.coords[i]-point_b.coords[i])**2;
        }

        result = result**0.5
        return result;
    }

    manhattan = (point_a, point_b)=>{
        let result = 0;
        for(let i = 0; i < point_a.coords.length; i++) {
            result += Math.abs(point_a.coords[i]-point_b.coords[i]);
        }

        return result;
    }
}

class FuzzyKNN extends KNN {
    fit = ({pointsToFit, k, distance_method, m})=>{

        if(m == 1) {
            throw("Parameter 'm' cannot be equal to 1 !");
        }
        const results = {k: k, points: {}};
        for(let i = 0; i < pointsToFit.length; i++) {
            this.calc_distances(pointsToFit[i], distance_method);
            const nearest = this.get_k_nearest(k);
            const classes = [];
            let denominator = 0;
            const heaviest_class = {test_class: null, weight: 0, actual_class: pointsToFit[i].class};

            for(let j = 0; j < k; j++) {
                
                denominator += (1/(nearest[j].distance || Number.EPSILON)**(2/(m-1)));

                if(!classes.includes(nearest[j].class)) {
                    classes.push(nearest[j].class);
                }
            }
            
            for(let j = 0; j < classes.length; j++) {
                let numerator = 0;

                for(let l = 0; l < k; l++) {
                    if(nearest[l].class !== classes[j]) {
                        continue;
                    } else {
                        numerator += (1/((nearest[l].distance || Number.EPSILON)**(2/(m-1))));
                    }        
                }

                const weight = (numerator/denominator);

                
                if(weight > heaviest_class.weight) {
                    heaviest_class.test_class = classes[j];
                    heaviest_class.weight = weight;
                }
            }

            results.points[pointsToFit[i].coords] = heaviest_class;
        }
        return results;
    }
}



module.exports = {KNN, FuzzyKNN, distances};