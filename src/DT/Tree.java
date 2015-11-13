package DT;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Tree {

    private double[][] featureAttribute;
    private int featureSplit;

    public Tree() {

    }

    public void train (double[][] patterns, double[] labels, int featureSplit) {
        if(patterns.length != labels.length) {
            return;
        }

        this.featureSplit = featureSplit;
        computeFeatureAttribute(patterns);
    }

    private void computeFeatureAttribute(double[][] patterns) {
        featureAttribute = new double[patterns[0].length][featureSplit+1];
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        double splitSize;
        for (int i = 0; i < patterns[0].length; i++) {
            for (int j = 0; j < patterns.length; j++) {
                if (max < patterns[j][i]) {
                    max = patterns[j][i];
                }
                else if (min > patterns[j][i]) {
                    min = patterns[j][i];
                }
            }
            splitSize = (max - min)/((double) featureSplit);
            for(int j = 0; j <= featureSplit; j++) {
                if (j == 0) {
                    featureAttribute[i][j] = Double.NEGATIVE_INFINITY;
                } else if(j == featureSplit) {
                    featureAttribute[i][j] = Double.POSITIVE_INFINITY;
                }
                else {
                    featureAttribute[i][j] = min + splitSize * j;
                }
            }
        }
    }

    private double[][] computeSubLabels(double[][] patterns, double[] labels, int featureNumber) {
        double[][] subLabels = new double[featureSplit][];
        double lowerBound, upperBound;
        double[] selectedFeature = selectFeature(patterns, featureNumber);
        int count, index;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = featureAttribute[featureNumber][i];
            upperBound = featureAttribute[featureNumber][i+1];
            count = countHitValue(selectedFeature, lowerBound, upperBound);
            subLabels[i] = new double[count];
            index = 0;
            for (int j = 0; j < selectedFeature.length; j++) {
                if (lowerBound <= selectedFeature[j] && selectedFeature[j] < upperBound) {
                    subLabels[i][index] = labels [j];
                }
            }
        }
        return subLabels;
    }

    private List<double[][]> splitData(int featureNumber, double value, double[][] patterns, double[] labels) {
        List<double[][]> splitData = new LinkedList<>();
        double [] selectedFeature = selectFeature(patterns, featureNumber);
        int count = countHitValue(selectedFeature, Double.NEGATIVE_INFINITY, value);
        double[][] leftPatterns = new double[count][];
        double [][] leftLabels = new double[count][1];
        double[][] rightPatterns = new double[selectedFeature.length-count][];
        double [][] rightLabels = new double[selectedFeature.length-count][1];

        int countRight = 0, countLeft = 0;

        for (int i = 0; i < selectedFeature.length; i++) {
            if(selectedFeature[i] < value) {
                leftPatterns[countLeft] = patterns[i];
                leftLabels[countLeft][0] = labels[i];
                countLeft++;
            } else {
                rightPatterns[countRight] = patterns[i];
                rightLabels[countRight][0] = labels[i];
                countRight++;
            }
        }
        splitData.add(leftPatterns);
        splitData.add(leftLabels);
        splitData.add(rightPatterns);
        splitData.add(rightLabels);
        return splitData;
    }

    private double [] selectFeature(double[][] patterns, int featureNumber) {
        double[] selectedFeature = new double[patterns.length];
        for(int i = 0; i < patterns.length; i++) {
            selectedFeature[i] = patterns[i][featureNumber];
        }
        return selectedFeature;
    }

    private double[] computeInformationGain(double[] labels, double[][] subLabels) {
        double [] gain = new double [featureSplit];
        for (int i = 0; i < featureSplit; i++) {
            gain[i] = computeEntropy(labels);
            for (int j = 0; j < featureSplit; j++) {
                if(i == j) {
                    gain[i] -= (subLabels[j].length/labels.length)*computeEntropy(subLabels[j]);
                } else {
                    gain[i] += (subLabels[j].length/labels.length)*computeEntropy(subLabels[j]);
                }
            }
        }
        return gain;
    }

    private double computeEntropy(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int[] distribution = computeClassDistribution(labels);

        double entropy = Double.NEGATIVE_INFINITY;
        for (int i = 0; i <= maxLabel; i++) {
            entropy += -(distribution[i]/numberOfLabels)*(Math.log(distribution[i]/numberOfLabels)/Math.log(2.0));
        }
        return entropy;
    }

    private int computeMaxLabel(double[] labels) {
        int maxLabel = Integer.MIN_VALUE;
        int numberOfLabels = labels.length;
        for (int i = 0; i < numberOfLabels; i++) {
            if (maxLabel < (int) labels[i]) {
                maxLabel = (int) labels[i];
            }
        }
        return maxLabel;
    }

    private int[] computeClassDistribution(double [] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        for (int i = 0; i < numberOfLabels; i++){
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }

    private int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (lowerBound <= selectedFeature[i] && selectedFeature[i] < upperBound) {
                count++;
            }
        }
        return count;
    }
}
