package KNN;

/**
 * K-Nearest-Neighbor classifier (own implementation)
 */

public class KNN {

    /**
     * Label- and feature-set
     */
    private double[][] trainPatterns;
    private double[] trainLabels;


    /**
     * Method classifies the feature-set with a given distance Calculation
     * @param knn Number of neighbors
     * @param patterns Feature-set
     * @param distanceCalculation Distance function
     * @return classified label-set
     */
    public double[] classify( int knn, double[][] patterns,String distanceCalculation){
        double [] predictedLabel = new double[patterns.length];
        for (int i = 0; i < predictedLabel.length; i++) {
            predictedLabel[i] = classify(knn, patterns[i],distanceCalculation);

        }
        return predictedLabel;
    }

    /**
     * Methods trains the classifier with the given label- and feature-set
     * @param patterns Feature-set
     * @param labels Label-set
     */
    public void train(double[][] patterns, double[] labels){
        trainPatterns = new double[patterns.length][patterns[0].length];
        trainLabels = new double[labels.length];

        for (int i = 0; i < patterns.length; i++) {
            trainLabels[i] = labels[i];
            for (int j = 0; j < trainPatterns[0].length; j++) {
                trainPatterns[i][j] = patterns[i][j];
            }
        }
    }


    /**
     * Method classifies the individual features (timestamps).
     * @param knn Number of Neighbors
     * @param pattern Feature-vector
     * @param distanceCalculation Distance function
     * @return classified label
     */
    private int classify(int knn, double[] pattern,String distanceCalculation) {
        double[][] dataPattern = trainPatterns;
        double[] label = trainLabels;
        double[] distance = new double[label.length];
        if(distanceCalculation.equals("Euclidean")){
            for (int i = 0; i < label.length; i++){
                double squaresum = 0;
                for (int j = 0; j < pattern.length; j++) {
                    squaresum += Math.pow((dataPattern[i][j] - pattern[j]), 2);
                }
                distance[i] = Math.sqrt(squaresum);
            }
        }else if (distanceCalculation.equals("Manhattan")){
                for (int i = 0; i < label.length; i++){
                    double squaresum = 0;
                    for (int j = 0; j < pattern.length; j++) {
                        squaresum += Math.abs((dataPattern[i][j] - pattern[j]));
                    }
                    distance[i] = squaresum;
                }
        }

        double[] distanceCheck = distance;
        int[] extrema = new int[knn];
        for (int t = 0; t < knn; t++) {
            for (int z = 0; z < distanceCheck.length; z++) {
                for (int w = z; w < distanceCheck.length; w++) {
                    if (distanceCheck[z] > distanceCheck[w]) {
                        z = w - 1;
                        break;
                    } else if (w == distanceCheck.length - 1) {
                        extrema[t] = z;
                        distanceCheck[z] = Integer.MAX_VALUE;
                        z = distanceCheck.length + 1;
                        break;
                    }
                }
            }
        }
        int[] labelArray = new int[extrema.length];
        for (int f = 0; f < extrema.length; f++) {
            labelArray[f] = (int) label[extrema[f]];
        }
        int singleLabel = getPopularElement(labelArray);

        return singleLabel;
    }

    /**
     * Method returns the popular element.
     * @param a Int-array
     * @return Most popular element
     */
    private int getPopularElement(int[] a) {
        int count = 1, tempCount;
        int popular = a[0];
        int temp = 0;
        for (int i = 0; i < (a.length - 1); i++) {
            temp = a[i];
            tempCount = 0;
            for (int j = 1; j < a.length; j++) {
                if (temp == a[j])
                    tempCount++;
            }
            if (tempCount > count) {
                popular = temp;
                count = tempCount;
            }
        }
        return popular;
    }
}