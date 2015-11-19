package DT;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    private int featureSplit;
    private Node root;
    private List<Integer> usedFeature = new LinkedList<>();
    private List<Node> leafs = new LinkedList<>();
    private double[][] quantifyValues;
    private int defaultLabel;
    private double[] values = {Double.NEGATIVE_INFINITY, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};
    /**
     * Methode legt einen neuen DT an und trainiert ihn mit den übergebenen Daten
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param deep Maximale Tiefe des Baums
     */
    public void train (double[][] patterns, double[] labels, int deep) {
        this.featureSplit = featureSplit;
        patterns = standardization(patterns);
        featureSplit = values.length - 1;
        //quantifyData(patterns);
        double[][] merge = merge(patterns, labels);
        defaultLabel = computeStrongestLabel(labels);
        root = build(transpose(merge), null, featureSplit, deep);
    }

    private double[][] standardization(double[][] patterns) {
        double[] median = new double[patterns[0].length];
        double[] standardDeviation = new double[patterns[0].length];
        double[] min = new double[patterns[0].length];
        double[] max = new double[patterns[0].length];
        double samples = (double)patterns.length;

        for (int i = 0; i < patterns[0].length; i++) {
            min[i] = Double.POSITIVE_INFINITY;
            max[i] = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < patterns.length; j++) {
                if(min[i] > patterns[j][i]) {
                    min[i] = patterns[j][i];
                }
                if (max[i] < patterns[j][i]) {
                    max[i] = patterns[j][i];
                }
                median[i] += patterns[j][i];
            }
            median[i] = median[i]/samples;
            for (int j = 0; j < patterns.length; j++) {
                standardDeviation[i] += Math.pow(patterns[j][i]-median[i], 2);
            }
            standardDeviation[i] = Math.sqrt(standardDeviation[i]/samples);

            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = (patterns[j][i] - median[i])/standardDeviation[i];
            }

        }
        return patterns;
    }



    public Node build (double[][] data, Node parent, int children, int deep) {
        Node node = new Node();
        node.parent = parent;
        node.children = new Node[children];

        if(data[0].length == 0) {
            System.out.println("Leaf (patterns = 0)");
            node.setLeaf(true);
            node.setClassLabel((double) defaultLabel);
            leafs.add(node);
            return node;
        }
        else if(deep <= 0) {
            System.out.println("Leaf (deep = 0)");
            int[] distribution = computeClassDistribution(data);
            int maxDistribution = Integer.MIN_VALUE;
            int  maxLabel = 0;
            for (int i = 0; i < distribution.length; i++) {

                if (maxDistribution < distribution[i]) {
                    maxDistribution = distribution[i];
                    maxLabel = i;
                }
            }
            node.setLeaf(true);
            node.setClassLabel(maxLabel);
            node.children = null;
            leafs.add(node);
            return node;
        }
        else if(isNodePure(data[data.length-1])) {
            System.out.println("Leaf (pure)");

            node.setLeaf(true);
            node.setClassLabel(data[data.length-1][0]);
            node.children = null;
            leafs.add(node);
            return node;
        }
        else {
            System.out.println("Node");
            node.setLeaf(false);
            double maxIG = Double.NEGATIVE_INFINITY, ig;
            int maxIGFeature = Integer.MIN_VALUE;
            for(int i = 0; i < data.length -1; i++) {
                ig = computeInformationGain(data, i);
                if(ig > maxIG /*&& !usedFeature.contains(i)*/) {
                    maxIG = ig;
                    maxIGFeature = i;
                }
            }
            usedFeature.add(maxIGFeature);

            deep--;
            node.setDecisionAttribute(maxIGFeature);
            node.setDecisionValues(values);

            System.out.println("Selected Feature: " + maxIGFeature + " IG: " + maxIG + " Deep: " + deep);
            double[][][] newData = splitData(data, maxIGFeature);

            for (int i = 0; i < featureSplit; i++) {
                System.out.println("Verteilung: " + newData[i][0].length);
            }

            for (int i = 0; i < featureSplit; i++) {
                node.children[i] = build(newData[i], node, children, deep);
            }

            return node;
        }
    }


    private int[] computeClassDistribution(double[][] data) {
        int numberOfLabels = data[0].length;
        int maxLabel = computeMaxLabel(data[data.length-1]);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)data[data.length-1][i]]++;
        }
        return distribution;
    }

    private int[] computeClassDistribution(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }


    private double[][] merge(double[][] patterns, double[] labels) {
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


    private double[][] sort(double[][] data, int featureNumber) {
        double[][] transpose = transpose(data);
        Arrays.sort(transpose, (double1, double2) -> {
            Double numOfKeys1 = double1[featureNumber];
            Double numOfKeys2 = double2[featureNumber];
            return numOfKeys1.compareTo(numOfKeys2);
        });
        data = transpose(transpose);
        return data;
    }

    private double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }

        }
        return transpose;
    }

    private void quantifyData(double[][] patterns) {
        quantifyValues = new double[patterns[0].length][featureSplit+1];
        double max, min;
        double quantify;
        for (int i = 0; i < patterns[0].length; i++) {
            max = Double.NEGATIVE_INFINITY;
            min = Double.POSITIVE_INFINITY;
            for (int j = 0; j < patterns.length; j++) {
                if(patterns[j][i] < min) {
                    min = patterns[j][i];
                }
                if(patterns[j][i] > max) {
                    max = patterns[j][i];
                }
            }
            quantify = (max - min)/((double)featureSplit);
            for (int j = 0; j <= featureSplit; j++) {
                if(j == 0) {
                    quantifyValues[i][j] = Double.NEGATIVE_INFINITY;
                }
                else if (j == featureSplit) {
                    quantifyValues[i][j] = Double.POSITIVE_INFINITY;
                }
                else {
                    quantifyValues[i][j] = min + quantify*(double)j;
                }
            }
        }
    }

    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[featureSplit][];
        double upperBound, lowerBound;
        int count, offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            //upperBound = quantifyValues[featureNumber][i+1];
            //lowerBound = quantifyValues[featureNumber][i];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j+offset];
            }
            offset += count;
        }
        return subLabels;
    }

    private double computeInformationGain(double[][] data, int featureNumber) {
        double[] labels = data[data.length-1];
        double[][] subLabels;
        double probability;

        double gain = computeEntropy(labels);
        subLabels = computeSubLabels(data, featureNumber);
        for (int j = 0; j < featureSplit; j++) {
            probability = ((double)subLabels[j].length)/((double)labels.length);
            gain -= (probability) * computeEntropy(subLabels[j]);
        }
        return gain;
    }

    /**
     * Methode zur Berechnung der Entropy
     * @param labels Labels/Sublabels auf den die Entropy berechnet werden soll
     * @return gibt den Entropywert zurueck
     */
    private double computeEntropy(double[] labels) {
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

    private double[][][] splitData(double[][] data, int featureNumber) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[featureSplit][][];
        int[] distribution = new int[featureSplit];
        double upperBound, lowerBound;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            //lowerBound = quantifyValues[featureNumber][i];
            //upperBound = quantifyValues[featureNumber][i+1];
            distribution[i] = countHitValue(data[featureNumber], lowerBound, upperBound);
        }
        int offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            splitData[i] = new double[data.length][distribution[i]];
            for (int j = 0; j < data.length; j++) {
                for(int k = 0; k < distribution[i]; k++) {
                    splitData[i][j][k] = dataSort[j][k + offset];
                }
            }
            offset += distribution[i];
        }

        return splitData;
    }


    /**
     * Methode sucht das maximale Label, dass in Labels/Sublabels vorkomm
     * @param labels Labels/Sublabels auf den das maximale Label gesucht wird
     * @return maximale Label in Labels/Sublabels
     */
    private int computeMaxLabel(double[] labels) {
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
     * Methode berechnet die Verteilung der Labels
     * @param labels Label/Sublabel, auf der die Verteilung berechnet wird.
     * @return int-Array mit der Labelverteilung
     */
    private int computeStrongestLabel (double [] labels) {
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
    private int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
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
    private boolean isNodePure(double[] labels) {
        for (int i = 0; i < labels.length; i++) {
            if (labels[0] !=labels[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Getter-Methode für Root-Knoten
     * @return Root Knoten
     */
    public Node getRoot() {
        return root;
    }

    /**
     * Methode klassifiziert die uebergebenen Patterns
     * @param patterns Patterns die klassifiziert Werden
     * @return double-Array mit den jeweiligen Labels
     */
    public double[] classify(double[][] patterns) {
        patterns = standardization(patterns);
        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }

    /**
     * Methode geht durch den Baum durch und liefert die jeweilige Klasse zurück wenn es auf ein Blatt trifft
     * @param pattern zu klassifizierendes Pattern
     * @return klassifiziertes Label
     */
    public double passTree(double[] pattern) {
        Node node = root;
        double classified, upperBound, lowerBound;
        double[] values;
        int feature;
        while (node.getLeaf() == false) {
            feature = node.getDecisionAttribute();
            values = node.getDecisionValues();
            for (int i = 0; i < featureSplit; i++) {
                lowerBound = values[i];
                upperBound = values[i+1];
                if(lowerBound <= pattern[feature] && pattern[feature] < upperBound) {
                    node = node.children[i];
                }
            }
        }
        classified = node.getClassLabel();
        return classified;
    }

}
