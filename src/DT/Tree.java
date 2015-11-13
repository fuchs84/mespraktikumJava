package DT;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Tree {

    private double[][] featureAttribute;
    private int featureSplit;

    /**
     * Methode zum Trainieren des DTs
     * @param patterns Train-Pattern zum trainieren des DTs
     * @param labels Train-Labels zum trainieren des DTs
     * @param featureSplit Teilt die Features in Bereiche ein
     */
    public void train (double[][] patterns, double[] labels, int featureSplit) {
        if(patterns.length != labels.length) {
            return;
        }

        this.featureSplit = featureSplit;
        computeFeatureAttribute(patterns);
    }


    private void learn() {

    }

    /**
     * Methode teilt die einzelnen Features in Bereiche ein und speichert sie in featureAttribute ab.
     * @param patterns Train-Patterns
     */
    private void computeFeatureAttribute(double[][] patterns) {
        featureAttribute = new double[patterns[0].length][featureSplit-1];
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
            for(int j = 1; j < featureSplit; j++) {
                featureAttribute[i][j-1] = min + splitSize * j;
            }
        }
    }

    /**
     * Methode trennt die Labels in SubLabels auf zum Berechnen des Information Gains
     * @param patterns Train-Patterns zur Entscheidung der SubLevel Einteilung
     * @param labels Train-Patterns zur Einteilung in SubLabels
     * @param featureNumber Ausgewähltes Feature nachdem SubLabels getrennt wird.
     * @return Sublabels
     *
     * Fehler Drinnen!!!!!!!!! FEATUREATTRIBUTE Initalisierung überprüfen!!!!!
     *
     */
    private double[][] computeSubLabels(double[][] patterns, double[] labels, int featureNumber) {
        double[][] subLabels = new double[featureSplit][];
        double upperBound;
        double[] selectedFeature = selectFeature(patterns, featureNumber);
        int count, index;
        for (int i = 0; i < featureSplit; i++) {
            upperBound = featureAttribute[featureNumber][i];
            count = countHitValue(selectedFeature, upperBound);
            subLabels[i] = new double[count];
            index = 0;
            for (int j = 0; j < selectedFeature.length; j++) {
                if (selectedFeature[j] < upperBound) {
                    subLabels[i][index] = labels [j];
                    index++;
                }
            }
        }
        return subLabels;
    }

    /**
     *
     * @param featureNumber
     * @param value
     * @param patterns
     * @param labels
     * @return
     */
    private List<double[][]> splitData(int featureNumber, double value, double[][] patterns, double[] labels) {
        List<double[][]> splitData = new LinkedList<>();
        double [] selectedFeature = selectFeature(patterns, featureNumber);
        int count = countHitValue(selectedFeature, value);
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

    /**
     * Methode selektiert ein ausgewähltes Feature aus den Train-Patterns
     * @param patterns Train-Patterns
     * @param featureNumber Ausgewähltes Feature
     * @return Selektiertes Feature aus den Train-Patterns
     */
    private double [] selectFeature(double[][] patterns, int featureNumber) {
        double[] selectedFeature = new double[patterns.length];
        for(int i = 0; i < patterns.length; i++) {
            selectedFeature[i] = patterns[i][featureNumber];
        }
        return selectedFeature;
    }

    /**
     *
     * @param labels
     * @param subLabels
     * @return
     */
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

    /**
     *
     * @param labels
     * @return
     */
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

    /**
     *
     * @param labels
     * @return
     */
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

    /**
     *
     * @param labels
     * @return
     */
    private int[] computeClassDistribution(double [] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        for (int i = 0; i < numberOfLabels; i++){
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }

    /**
     *
     * @param selectedFeature
     * @param upperBound
     * @return
     */
    private int countHitValue(double [] selectedFeature, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (selectedFeature[i] < upperBound) {
                count++;
            }
        }
        return count;
    }
}
