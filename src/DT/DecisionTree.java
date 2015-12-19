package DT;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.util.Arrays;
import java.util.Comparator;
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
    protected List<Double> usedLabels = new LinkedList<>();
    protected int numberOfInstances;
    protected int deep;
    protected int minNodeSize;
    protected Matrix pcaMatrix;


    /**
     * Methode berechnet von allen Features die Standardisierung
     * @param patterns Train-Patterns
     * @return standardisierte Train-Patterns
     */
    public double[][] standardization(double[][] patterns) {
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

    /**
     * Methode berechnet die Kovarianzmatrix
     * @param patterns Train-Patterns
     * @return Kovarianzmatrix
     */
    protected double[][] computeCovarianceMatrix(double[][] patterns) {
        double samples = patterns.length;
        double[][] covariance = new double[patterns[0].length][patterns[0].length];
        double median;
        for(int i = 0; i < patterns[0].length; i++) {
            median = 0;
            for(int j = 0; j < patterns.length; j++) {
               median += patterns[j][i];
            }
            median = median/samples;
            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = patterns[j][i] - median;
            }
        }
        for(int j = 0; j < patterns[0].length; j++) {
            for (int k = 0; k <= j; k++) {
                for(int l = 0; l < patterns.length; l++) {
                    covariance[j][k] += patterns[l][j]*patterns[l][k];
                }
                covariance[k][j] = covariance[j][k] = covariance[j][k]/samples;

            }
        }
        return covariance;
    }

    public double[][] computePCA(double[][] patterns, int k) {
        double[][] covariance = computeCovarianceMatrix(patterns);
        Matrix covarianceMatrix = new Matrix(covariance);
        EigenvalueDecomposition evD = new EigenvalueDecomposition(covarianceMatrix);



        double [][] ev = evD.getV().getArray();


        double[][] pca = new double[ev.length][k];
        int evIndex = ev[0].length - 1;
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < ev.length; j++) {
                pca[j][i] = ev[j][evIndex];
            }
            evIndex--;
        }
        patterns = computeZeroMeanPatterns(patterns);
        Matrix patternsMatrix = new Matrix(patterns);
        pcaMatrix = new Matrix(pca);

        return (patternsMatrix.times(pcaMatrix).getArray());
    }

    public double[][] usePCA(double[][] patterns) {
        patterns = computeZeroMeanPatterns(patterns);
        Matrix patternsMatrix = new Matrix(patterns);
        return (patternsMatrix.times(pcaMatrix).getArray());
    }

    public double[][] computeZeroMeanPatterns(double[][] patterns) {
        int numberOfInstances = patterns.length;
        double mean;
        for(int i = 0; i < patterns[0].length; i++) {
            mean = 0.0;
            for(int j = 0; j < patterns.length; j++) {
               mean = mean + patterns[j][i];
            }
            mean = mean/(double)numberOfInstances;
            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = patterns[j][i] - mean;
            }
        }
        return patterns;
    }

    public double[][] normalisation(double[][] patterns) {
        double[][] newPatterns = new double[patterns.length][patterns[0].length];
        double max, min;
        for(int i = 0; i < patterns[0].length; i++) {
            max = Double.NEGATIVE_INFINITY;
            for(int j = 0; j < patterns.length; j++) {
                if(max < Math.abs(patterns[j][i])) {
                    max = Math.abs(patterns[j][i]);
                }

            }
            for(int j = 0; j < patterns.length; j++) {
                newPatterns[j][i] = patterns[j][i]/max;
            }
        }
        return newPatterns;
    }

    protected void searchUsedLabels(double[] labels) {
        for(int i = 0; i < labels.length; i++) {
            if(!(usedLabels.contains(labels[i]))) {
                usedLabels.add(labels[i]);
            }
        }
    }

    /**
     * Methode berechnet die quantifizierten Grenzwerte
     * @param data Train-Daten (Patterns + Labels)
     * @return quantifizierten Grenzwerte der einzelnen Features
     */
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
    protected double[][] sort(double[][] data, final int featureNumber) {
        double[][] transpose = transpose(data);
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
     * Methode berechnet die Entropie
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
     * Methode sucht das maximale Label
     * @param labels Train-Labels
     * @return maximale Label
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
     * Methode zaehlt die Sampels zwischen zwei Grenzen, die den Wert erfuellen
     * @param selectedFeature ausgewaeltes Feature
     * @param lowerBound untere Grenze
     * @param upperBound obere Grenze
     * @return Anzahl der Samples, die die Bedingung erfuellen
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
     * Methode ueberprueft, ob ein Konten pure ist
     * @param labels Train-Label eines Knotens
     * @return true, wenn der Knoten pure ist, andernfalls false
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
