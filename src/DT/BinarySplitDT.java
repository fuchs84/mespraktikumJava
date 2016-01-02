package DT;

import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 29.11.15.
 */
public class BinarySplitDT extends DecisionTree{

    private BinarySplitNode root;
    private int count;
    private int deep;



    public void train (double[][] patterns, double[] labels, int deep, int minNodeSize, int quantifySize) {
        this.minNodeSize = minNodeSize;
        patterns = normalisation(patterns);

        double[][] merge = merger(patterns, labels);
        double[][] data = transpose(merge);
        this.deep = deep;

        numberOfInstances = labels.length;
        this.quantifySize = quantifySize;
        defaultLabel = computeStrongestLabel(labels);
        entireDistribution = computeClassDistribution(labels);

        root = build(data, null, 0);
    }

    /**
     * Methode baut entweder einen neuen Knoten oder ein Blatt
     * @param data Train-Daten (Patterns + Labels)
     * @param parent Elternknoten
     * @param deep Tiefe des Baums
     * @return Knoten oder Blatt
     */
    public BinarySplitNode build (double[][] data, BinarySplitNode parent, int deep) {
        BinarySplitNode node = new BinarySplitNode();
        node.parent = parent;
        node.deep = deep;
//        if(data[0].length == 0) {
//            node.setLeaf(true);
//            int label = defaultLabel;
//            for(int i = 0; i < entireDistribution.length; i++) {
//                if((double) entireDistribution[i]/(double) numberOfInstances > Math.random() && i != defaultLabel) {
//                    label = i;
//                }
//            }
//            node.setClassLabel((double) label);
//
//            return node;
//        }
//        else
        if(isNodePure(data[data.length-1])) {
            node.setLeaf(true);
            node.setClassLabel(data[data.length - 1][0]);
            node.left = null;
            node.right = null;
            return node;
        }
        else if(this.deep <= deep || minNodeSize > data[0].length) {
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
            node.left = null;
            node.right = null;
            return node;
        }
        else {
            node.setLeaf(false);
            if(quantifySize < 2) {
                quantifySize--;
            }

            double[][] quantifyValues = computeQuantifyValues(data);
            double [][] entropyImpurity = new double[data.length-1][];


            for(int i = 0; i < entropyImpurity.length; i++) {
                entropyImpurity[i] = computeEntropyImpurity(data, quantifyValues, i);
            }

            double minImpurity = Double.POSITIVE_INFINITY;
            double minImpurityValue = Double.NEGATIVE_INFINITY;
            double minImpuritySplitDistribution = Double.NEGATIVE_INFINITY;
            int minImpurityFeature = Integer.MIN_VALUE;
            for(int i = 0; i < entropyImpurity.length; i++) {
                if(minImpurity >= entropyImpurity[i][0] && minImpuritySplitDistribution < entropyImpurity[i][3]) {
                    minImpurity = entropyImpurity[i][0];
                    minImpurityValue = entropyImpurity[i][1];
                    minImpurityFeature = (int) entropyImpurity[i][2];
                    minImpuritySplitDistribution = entropyImpurity[i][3];
                }
            }

            usedFeature.add(minImpurityFeature);

            node.setDecisionValueBound(minImpurityValue);
            node.setDecisionAttribute(minImpurityFeature);
            deep++;
            double[][][] newData = splitData(data, minImpurityFeature, minImpurityValue);

            node.left = build(newData[0], node, deep);
            node.right = build(newData[1], node, deep);
        }
        return node;
    }

    /**
     * Methode berechnet die Sublabels eines ausgewaehlten Features und Wert
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @param values ausgewaehlter Wert
     * @return Sublabels
     */
    private double[][] computeSubLabels(double[][] data, int featureNumber, double[] values) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[quantifySize][];
        double upperBound, lowerBound;
        int count;
        lowerBound = values[0];

        for (int i = 0; i < quantifySize; i++) {
            upperBound = values[i+1];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j];

            }
        }
        return subLabels;
    }

    /**
     * Methode berechnet die minimale Entropie-Unreinheit eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param quantifyValues quantifizierte Grenzwerte
     * @param featureNumber ausgewaehltes Feature
     * @return Entropie-Unreinheit und den dazugehörtigen Wert
     */
    private double[] computeEntropyImpurity(double[][] data, double[][] quantifyValues, int featureNumber) {
        double[] impurityAndValue = new double[4];
        double[][] subLabels = computeSubLabels(data, featureNumber, quantifyValues[featureNumber]);
        double[] impurity = new double[quantifySize];
        double splitDistribution = 0;
        for (int i = 0; i < quantifySize; i++) {
            if(subLabels[i].length > 0 && subLabels[i].length < data[0].length) {
                impurity[i] = computeEntropy(subLabels[i]);
            } else {
                impurity[i] = Double.POSITIVE_INFINITY;
            }

        }
        double minImpurity = Double.POSITIVE_INFINITY;
        double value = quantifyValues[featureNumber][0];
        for (int i = 0; i < quantifySize; i++) {
            if(minImpurity > impurity[i]) {
                minImpurity = impurity[i];
                value = quantifyValues[featureNumber][i+1];

                if(subLabels[i].length > (data[0].length-subLabels[i].length)) {
                    splitDistribution = (double)(data[0].length - subLabels[i].length)/(double)subLabels[i].length;
                } else {
                    splitDistribution = (double)subLabels[i].length/(double)(data[0].length - subLabels[i].length);
                }
            }
        }
        impurityAndValue[0] = minImpurity;
        impurityAndValue[1] = value;
        impurityAndValue[2] = featureNumber;
        impurityAndValue[3] = splitDistribution;
        return impurityAndValue;
    }

    /**
     * Methode teilt die Daten nach ein ausgewaehltes Feature auf
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @param value ausgewaehlter Wert
     * @return Array aus Train-Daten (Patterns + Labels)
     */
    private double[][][] splitData(double[][] data, int featureNumber, double value) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[2][][];
        int distribution;
        distribution = countHitValue(data[featureNumber], Double.NEGATIVE_INFINITY, value);

        splitData[0] = new double[data.length][distribution];
        splitData[1] = new double[data.length][data[0].length-distribution];

        for (int i = 0; i < data.length; i++) {
            for(int j = 0; j < distribution; j++) {
                splitData[0][i][j] = dataSort[i][j];
            }
            for(int j = 0; j < data[0].length-distribution; j++) {
                splitData[1][i][j] = dataSort[i][j + distribution];
            }
        }
        return splitData;
    }

    /**
     * Methode klassifiziert die uebergebenen Patterns
     * @param patterns Patterns die klassifiziert Werden
     * @return double-Array mit den jeweiligen Labels
     */
    public double[] classify(double[][] patterns) {

        patterns = normalisation(patterns);

        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }

    /**
     * Methode geht durch den Baum durch und liefert die jeweilige Klasse zurueck wenn es auf ein Blatt trifft
     * @param pattern klassifizierendes Pattern
     * @return klassifiziertes Label
     */
    public double passTree(double[] pattern) {
        BinarySplitNode binarySplitNode = root;
        double classified;

        int feature;
        double value;
        while (binarySplitNode.getLeaf() == false) {
            feature = binarySplitNode.getDecisionAttribute();
            value = binarySplitNode.getDecisionValueBound();
            if (pattern[feature] < value) {
                binarySplitNode = binarySplitNode.left;
            }
            else {
                binarySplitNode = binarySplitNode.right;
            }
        }
        classified = binarySplitNode.getClassLabel();
        return classified;
    }

    public void saveData() {
        try {
            FileWriter fw = new FileWriter("binarySplitDT.csv");
            save(root, fw);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void save(BinarySplitNode node, FileWriter fw) {
        try {
            if(node.getLeaf() == false) {
                fw.append(Boolean.toString(false));
                fw.append(",");
                fw.append(Integer.toString(node.getDecisionAttribute()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append(",");
                fw.append(Double.toString(node.getDecisionValueBound()));
                fw.append(",");
                fw.append("\n");
                save(node.left, fw);
                save(node.right, fw);
            }
            else {
                fw.append(Boolean.toString(true));
                fw.append(",");
                fw.append(Double.toString(node.getClassLabel()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadData() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("binarySplitDT.csv"));
            ArrayList<String> data = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                data.add(line);
            }
            count = - 1;
            this.root = buildWithLoadedData(null, data);
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BinarySplitNode buildWithLoadedData(BinarySplitNode parent, ArrayList<String> data) {

        count++;
        BinarySplitNode node = new BinarySplitNode();
        node.parent = parent;
        String[] parts = data.get(count).split(",");
        if(parts[0].equals("false")) {
            node.setLeaf(false);
            node.setDecisionAttribute(Integer.parseInt(parts[1]));
            node.setDecisionValueBound(Double.parseDouble(parts[3]));
            node.left = buildWithLoadedData(node, data);
            node.right = buildWithLoadedData(node, data);
        } else {
            node.setLeaf(true);
            node.setClassLabel(Double.parseDouble(parts[1]));
            node.left = null;
            node.right = null;
        }
        return node;
    }
}
