package DT;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    protected int featureSplit;
    protected List<Integer> usedFeature = new LinkedList<>();
    protected int defaultLabel;
    protected int[] entireDistribution;

    protected int numberOfInstances;


    /**
     * Methode berechnet von allen Features die Standardisierung
     * @param patterns Train-Patterns
     * @return standardisierte Train-Patterns
     */
    protected double[][] standardization(double[][] patterns) {
        double samples = (double)patterns.length;
        double median;
        double standardDeviation;
        for (int i = 0; i < patterns[0].length; i++) {
            median = 0;
            standardDeviation = 0;
            for (int j = 0; j < patterns.length; j++) {
                median += patterns[j][i];
            }
            median= median/samples;
            for (int j = 0; j < patterns.length; j++) {
                standardDeviation += Math.pow(patterns[j][i]-median, 2);
            }
            standardDeviation = Math.sqrt(standardDeviation/samples);

            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = (patterns[j][i] - median)/standardDeviation;
            }

        }
        return patterns;
    }


    protected double[][] computeCovarianceMatrix(double[][] data) {
        double samples = data[0].length;
        double[][] covariance = new double[data.length-1][];
        double median;
        for(int i = 0; i < data.length-1; i++) {
            median = 0;
            for(int j = 0; j < data[0].length; j++) {
               median += data[i][j];
            }
            median = median/samples;
            for(int j = 0; j < data[0].length; j++) {
                data[i][j] = data[i][j] - median;
            }
        }
        for(int j = 0; j < data.length-1; j++) {
            covariance[j] = new double[j+1];
            for (int k = 0; k <= j; k++) {
                for(int l = 0; l < data[0].length; l++) {
                    covariance[j][k] += data[j][l]*data[k][l];
                }
                covariance[j][k] = covariance[j][k]/samples;
            }
        }
        return covariance;
    }

    protected double[][] computeQuantifyValues(double[][] data) {
        double [][] quantifyValues = new double[data.length-1][featureSplit+1];
        double max, min, quantify;
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
            quantify = (max - min)/((double) featureSplit);
            for(int j = 0; j < featureSplit+1; j++) {
                if(j == 0) {
                    quantifyValues[i][j] = Double.NEGATIVE_INFINITY;
                }
                else if(j == featureSplit) {
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
     * Methode berechnet die Klassenverteilung
     * @param data Train-Daten (Patterns + Labels)
     * @return Verteilung der einzelnen Labels
     */
    protected int[] computeClassDistribution(double[][] data) {
        int numberOfLabels = data[0].length;
        int maxLabel = computeMaxLabel(data[data.length-1]);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)data[data.length-1][i]]++;
        }
        return distribution;
    }

    /**
     * Methode berechnet die Klassenverteilung
     * @param labels Train-Labels
     * @return Verteilung der einzelnen Labels
     */
    protected int[] computeClassDistribution(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }

    /**
     * Methode fuegt Patterns und Labels zusammen
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @return Train-Daten (Patterns + Labels)
     */
    protected double[][] merger(double[][] patterns, double[] labels) {
        if (patterns.length != labels.length) {
            System.out.println("Patterns und Labels passen nicht zusammen!");
            return null;
        }
        double[][] merge = new double[patterns.length][patterns[0].length + 1];
        for (int i = 0; i < merge.length; i++) {
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
     * Methode sortiert die Daten nach nach einem ausgewaehlten Feature aufsteigend
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber Ausgewaehltes Feature
     * @return sortierte Train-Daten (Patterns + Labels)
     */
    protected double[][] sort(double[][] data, int featureNumber) {
        double[][] transpose = transpose(data);
        Arrays.sort(transpose, (double1, double2) -> {
            Double numOfKeys1 = double1[featureNumber];
            Double numOfKeys2 = double2[featureNumber];
            return numOfKeys1.compareTo(numOfKeys2);
        });
        data = transpose(transpose);
        return data;
    }

    /**
     * Methode transponiert die Daten (Matrix-Transposition)
     * @param data Train-Daten (Patterns + Labels)
     * @return transponierte Train-Daten (Patterns + Labels)
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
     * Methode zur Berechnung der Entropy
     * @param labels Labels/Sublabels auf den die Entropy berechnet werden soll
     * @return gibt den Entropywert zurueck
     */
    protected double computeEntropy(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int[] distribution = computeClassDistribution(labels);


        double entropy = 0.0;
        double probability;
        for (int i = 0; i <= maxLabel; i++) {

            if (distribution[i] != 0) {

                probability = ((double)distribution[i])/((double)numberOfLabels);
                entropy += -(probability) * (Math.log(probability) / Math.log(2.0));
            }
        }
        return entropy;
    }

    /**
     * Methode sucht das maximale Label, dass in Labels/Sublabels vorkomm
     * @param labels Labels/Sublabels auf den das maximale Label gesucht wird
     * @return maximale Label in Labels/Sublabels
     */
    protected int computeMaxLabel(double[] labels) {
        int maxLabel = 0;
        int numberOfLabels = labels.length;
        for (int i = 0; i < numberOfLabels; i++) {
            if (maxLabel < (int) labels[i]) {
                maxLabel = (int) (labels[i]);
            }
        }
        return maxLabel;
    }

    /**
     * Methode sucht das staerkste Label heraus
     * @param labels Train-Labels
     * @return staerkste Label
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
        for (int i = 0; i < distribution.length; i++) {
            if(distribution[i] > numberOfStrongestLabel) {
                numberOfStrongestLabel = distribution[i];
                strongestLabel = i;
            }
        }
        return strongestLabel;
    }

    /**
     * Methode zaehlt die Sampels, bis zur oberen Grenze, die den Wert erfuellen
     * @param selectedFeature ausgewaeltes Feature
     * @param upperBound obere Grenze
     * @return Anzahl der Sampels, die die Bedingung erfuellen
     */
    protected int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (lowerBound <= selectedFeature[i] && selectedFeature[i] < upperBound) {
                count++;
            }
        }
        return count;
    }

    /**
     * Methode ueberprueft, ob ein Konten/Subset pure ist
     * @param labels Train-Label-Subset eines Knotens
     * @return true, wenn der Knoten/Subset pure ist, andernfalls false
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
