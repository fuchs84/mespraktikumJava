package DT;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.io.*;
import java.util.*;

/**
 * Generally class for the decision tree
 */
public class DecisionTree {

    //Split size for calculating decision rule
    protected int quantifySize;

    //List of the used features
    protected List<Integer> usedFeature = new LinkedList<>();

    //Default label
    protected int defaultLabel;

    //Label distribution
    protected int[] entireDistribution;

    //List of the used labels
    protected List<Double> usedLabels = new LinkedList<>();

    //Number of instances
    protected int numberOfInstances;

    //Depth of the tree
    protected int deep;

    //Minimal node size
    protected int minNodeSize;



    /**
     * Method calculates the standardization of the features
     * @param patterns Feature-set
     * @return Standardised feature-set
     */
    public double[][] standardization(double[][] patterns) {
        double samples = (double)patterns.length;
        double median;
        double standardDeviation;
        for (int i = 0; i < patterns[0].length; i++) {
            median = 0;
            standardDeviation = 0;
            // sum up the feature values for each feature
            for (int j = 0; j < patterns.length; j++) {
                median += patterns[j][i];
            }
            // calculates the median
            median= median/samples;


            // calculates the standard deviation for the standardisation
            for (int j = 0; j < patterns.length; j++) {
                standardDeviation += Math.pow(patterns[j][i]-median, 2);
            }
            standardDeviation = Math.sqrt(standardDeviation/samples);

            // calculates the standardisation
            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = (patterns[j][i] - median)/standardDeviation;
            }

        }
        return patterns;
    }

    /**
     * Method calculates the normalisation of the features
     * @param patterns Feature-set
     * @return Normalised feature-set
     */
    public double[][] normalisation(double[][] patterns) {
        double[][] newPatterns = new double[patterns.length][patterns[0].length];
        double max, min;

        // search the minimum and maximum in each feature
        for(int i = 0; i < patterns[0].length; i++) {
            max = Double.NEGATIVE_INFINITY;
            min = Double.POSITIVE_INFINITY;
            for(int j = 0; j < patterns.length; j++) {
                if(max < patterns[j][i]) {
                    max =patterns[j][i];
                }
                if(min > patterns[j][i]) {
                    min = patterns[j][i];
                }
            }
            //calculates the normalisation for each feature
            for(int j = 0; j < patterns.length; j++) {
                newPatterns[j][i] = (patterns[j][i]-min)/(max-min);
            }
        }
        return newPatterns;
    }


    /**
     * Method calculates the quantifies Values for each feature.
     * @param data feature-set with labels
     * @return Quantified values for each feature
     */
    protected double[][] computeQuantifyValues(double[][] data) {
        double [][] quantifyValues = new double[data.length-1][quantifySize +1];
        double max, min, quantify;

        // search the minimum and maximum of each feature
        for(int i = 0; i < data.length-1; i++) {
            min = Double.POSITIVE_INFINITY;
            max = Double.NEGATIVE_INFINITY;
            for(int j = 0; j < data[i].length; j++) {
                if(data[i][j]  > max) {
                    max = data[i][j];
                }
                if(data[i][j] < min) {
                    min = data[i][j];
                }
            }

            // calculates the quantifies values (bound values) for each feature
            quantify = (max - min)/((double) quantifySize);
            for(int j = 0; j < quantifySize +1; j++) {
                if(j == 0) {
                    quantifyValues[i][j] = Double.NEGATIVE_INFINITY;
                }
                else if(j == quantifySize) {
                    quantifyValues[i][j] = Double.POSITIVE_INFINITY;
                }
                else {
                    quantifyValues[i][j] = min + (quantify*(double)j);
                }
            }
        }
        return quantifyValues;
    }

    /**
     * Method calculates the label distribution.
     * @param data Feature-set with labels
     * @return Label distribution
     */
    protected int[] computeClassDistribution(double[][] data) {
        int numberOfLabels = data[0].length;

        // search the maximal Label
        int maxLabel = computeMaxLabel(data[data.length-1]);
        int [] distribution = new int[maxLabel+1];

        // calculates the distribution of the labels
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)data[data.length-1][i]]++;
        }
        return distribution;
    }

    /**
     * Method calculates the label distribution.
     * @param labels Label-set
     * @return Label distribution
     */
    protected int[] computeClassDistribution(double[] labels) {
        int numberOfLabels = labels.length;

        // search the maximal Label
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        // calculates the distribution of the labels
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }


    /**
     * Method merges the feature-set and label-set
     * @param patterns Feature-set
     * @param labels Label-set
     * @return feature-set with labels
     */
    protected double[][] merger(double[][] patterns, double[] labels) {
        if (patterns.length != labels.length) {
            System.out.println("Patterns und Labels passen nicht zusammen!");
            return null;
        }
        double[][] merge = new double[patterns.length][patterns[0].length + 1];
        for (int i = 0; i < merge.length; i++) {
            //merges each label instance to the end of the feature instance
            for (int j = 0; j < merge[0].length; j++) {
                if(j == merge[0].length-1) {
                    merge[i][j] = labels[i];
                } else {
                    merge[i][j] = patterns[i][j];
                }
            }
        }
        return merge;
    }

    /**
     * Mothod sorts the data-set by a selected label.
     * @param data Feature-set with labels
     * @param featureNumber selected feature
     * @return Sorted Feature-set with labels
     */
    protected double[][] sort(double[][] data, final int featureNumber) {
        //transposes the data-set for sorting
        double[][] transpose = transpose(data);

        //sorts the data-set
        Arrays.sort(transpose, new Comparator<double[]>() {
            @Override
            public int compare(double[] double1, double[] double2) {
                Double numOfKeys1 = double1[featureNumber];
                Double numOfKeys2 = double2[featureNumber];
                return numOfKeys1.compareTo(numOfKeys2);
            }
        });
        data = transpose(transpose);
        return data;
    }


    /**
     * Method transpose a given matrix
     * @param data 2d-matrix (feature-set with labels)
     * @return Transposed matrix
     */
    protected double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }
        }
        return transpose;
    }

    /**
     * Method calculates the entropy
     * @param labels Label-set
     * @return Calculated entropy
     */
    protected double computeEntropy(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);

        //calculates the distribution of the label-set
        int[] distribution = computeClassDistribution(labels);

        double entropy = 0.0;
        double probability;
        for (int i = 0; i <= maxLabel; i++) {
            //calculates the entropy of the label-set
            if (distribution[i] != 0) {
                probability = ((double)distribution[i])/((double)numberOfLabels);
                entropy += -(probability) * (Math.log(probability) / Math.log(2.0));
            }
        }
        return entropy;
    }

    /**
     * Method searches the maximal label
     * @param labels Label-set
     * @return Maximal label
     */
    protected int computeMaxLabel(double[] labels) {
        int maxLabel = 0;
        int numberOfLabels = labels.length;

        //searches the maximal label by comparing each label with the current maximal label
        for (int i = 0; i < numberOfLabels; i++) {
            if (maxLabel < (int) labels[i]) {
                maxLabel = (int) (labels[i]);
            }
        }
        return maxLabel;
    }

    /**
     * Method searches the strongest label
     * @param labels Label-set
     * @return strongest label
     */
    protected int computeStrongestLabel (double [] labels) {
        int numberOfLabels = labels.length;
        int strongestLabel = Integer.MIN_VALUE;
        int numberOfStrongestLabel = Integer.MIN_VALUE;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }

        //searches the strongest label by comparing with the label-distributions
        for (int i = 0; i < distribution.length; i++) {
            if(distribution[i] > numberOfStrongestLabel) {
                numberOfStrongestLabel = distribution[i];
                strongestLabel = i;
            }
        }
        return strongestLabel;
    }

    /**
     * Method counts the samples between two bounds
     * @param selectedFeature Selected Feature
     * @param lowerBound Lower bound
     * @param upperBound Upper bound
     * @return Number of count
     */
    protected int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            //counts if the feature value in the bounds
            if (lowerBound <= selectedFeature[i] && selectedFeature[i] < upperBound) {
                count++;
            }
        }
        return count;
    }

    /**
     * Method checks the purity of a node
     * @param labels Label-set
     * @return True if a node is pure, else false
     */
    protected boolean isNodePure(double[] labels) {
        for (int i = 0; i < labels.length; i++) {
            if (labels[0] !=labels[i]) {
                return false;
            }
        }
        return true;
    }
}
