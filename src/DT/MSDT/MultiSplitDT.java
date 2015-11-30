package DT.MSDT;

import DT.DecisionTree;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 29.11.15.
 */
public class MultiSplitDT extends DecisionTree {

    private MultiSplitNode root;
    private List<MultiSplitNode> leafs = new LinkedList<>();
    private double[] values = {Double.NEGATIVE_INFINITY, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};


    public void train (double[][] patterns, double[] labels, int deep) {
        patterns = standardization(patterns);
        double[][] merge;
        double[][] data;

        merge = merger(patterns, labels);
        data = transpose(merge);

        numberOfInstances = labels.length;
        featureSplit = values.length - 1;
        defaultLabel = computeStrongestLabel(labels);
        entireDistribution = computeClassDistribution(labels);
        root = build(data, null, deep);
    }

    /**
     * Methode baut entweder einen neuen Knoten oder ein Blatt
     * @param data Train-Daten (Patterns + Labels)
     * @param parent Elternknoten
     * @param deep Tiefe des Baums
     * @return Knoten oder Blatt
     */

    public MultiSplitNode build (double[][] data, MultiSplitNode parent, int deep) {
        MultiSplitNode node = new MultiSplitNode();
        node.parent = parent;
        if(data[0].length == 0) {
            System.out.println("Leaf (patterns = 0)");
            node.setLeaf(true);
            int label = defaultLabel;
            for(int i = 0; i < entireDistribution.length; i++) {
                if((double) entireDistribution[i]/(double) numberOfInstances > Math.random() && i != defaultLabel) {
                    label = i;
                }
            }
            System.out.println("Label: " + label);
            node.setClassLabel((double) label);
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
            node.setLeaf(false);
            double maxIG = Double.NEGATIVE_INFINITY;
            int maxIGFeature = Integer.MIN_VALUE;
            double informationGain;
            for(int i = 0; i < data.length -1; i++) {
                informationGain = computeInformationGain(data, i);
                if(informationGain > maxIG && !usedFeature.contains(i)) {
                    maxIG = informationGain;
                    maxIGFeature = i;
                }
            }
            usedFeature.add(maxIGFeature);

            deep--;
            node.setDecisionAttribute(maxIGFeature);
            node.setDecisionValues(values);

            System.out.println("Selected Feature: " + maxIGFeature + " IG: " + maxIG + " Deep: " + deep);
            double[][][] newData = splitData(data, maxIGFeature);

            node.children = new MultiSplitNode[featureSplit];

            for (int i = 0; i < featureSplit; i++) {
                node.children[i] = build(newData[i], node, deep);
            }
            return node;
        }
    }

    /**
     * Methode teilt die Daten nach ein ausgewaehltes Feature auf
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Array aus Train-Daten (Patterns + Labels)
     */
    private double[][][] splitData(double[][] data, int featureNumber) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[featureSplit][][];
        int[] distribution = new int[featureSplit];
        double upperBound, lowerBound;


        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
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

    private double computeInformationGain(double[][] data, int featureNumber) {
        double[] labels = data[data.length-1];
        double[][] subLabels;
        double probability;
        double gain;
        subLabels = computeSubLabels(data, featureNumber);
        gain = computeEntropy(labels);
        for (int j = 0; j < featureSplit; j++) {
            probability = ((double)subLabels[j].length)/((double)labels.length);

            gain -= (probability) * computeEntropy(subLabels[j]);
        }
        return gain;
    }

    /**
     * Methode berechnet die Sublabels eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Sublabels
     */
    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[featureSplit][];
        double upperBound, lowerBound;
        int count, offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j+offset];
            }
            offset += count;
        }
        return subLabels;
    }

    /**
     * Getter-Methode für Root-Knoten
     * @return Root Knoten
     */
    public MultiSplitNode getRoot() {
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
        MultiSplitNode node = root;
        double classified, upperBound, lowerBound;

        int feature;
        double[] values;

        while (node.getLeaf() == false) {
            feature = node.getDecisionAttribute();
            values = node.getDecisionValues();
            for (int i = 0; i < featureSplit; i++) {
                lowerBound = values[i];
                upperBound = values[i + 1];
                if (lowerBound <= pattern[feature] && pattern[feature] < upperBound) {
                    node = node.children[i];
                }
            }
        }
        classified = node.getClassLabel();
        return classified;
    }
}
