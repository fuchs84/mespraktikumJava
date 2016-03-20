package DT;

import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;

/**
 * Binary-split decision Tree
 */
public class BinarySplitDT extends DecisionTree{

    // Root node
    private BinarySplitNode root;

    // Counter for depth
    private int count;


    /**
     * Method trains the classifier.
     * @param patterns feature-set
     * @param labels label-set
     * @param deep maximal depth
     * @param minNodeSize minimal numbers of instances in a node
     * @param quantifySize number of split parts for decision rules
     */
    public void train (double[][] patterns, double[] labels, int deep, int minNodeSize, int quantifySize) {

        this.minNodeSize = minNodeSize;

        //calculates the normalised feature-set
        patterns = normalisation(patterns);

        //merge label-set and feature set to data and afterwards transpose the data
        double[][] merge = merger(patterns, labels);
        double[][] data = transpose(merge);

        this.deep = deep;

        numberOfInstances = labels.length;
        this.quantifySize = quantifySize;

        //the strongest label is the default label
        defaultLabel = computeStrongestLabel(labels);

        //label distribution of the complete label-set
        entireDistribution = computeClassDistribution(labels);

        //builds the tree
        root = build(data, null, 0);
    }

    /**
     * Method builds recursive the decision tree
     * @param data Label + feature-set
     * @param parent parent node
     * @param deep current depth of the tree
     * @return node or leaf
     */
    public BinarySplitNode build (double[][] data, BinarySplitNode parent, int deep) {

        //new node
        BinarySplitNode node = new BinarySplitNode();
        node.parent = parent;
        node.deep = deep;

        // builds leaf if the instances of a node are pure
        if(isNodePure(data[data.length-1])) {
            node.setLeaf(true);
            node.setClassLabel(data[data.length - 1][0]);
            node.left = null;
            node.right = null;
            return node;
        }

        // builds leaf if maximal depth or minimal number of instances in a node
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

        // builds node with decision rule
        else {


            double[][] quantifyValues = computeQuantifyValues(data);
            double [][] entropyImpurity = new double[data.length-1][];

            // calculates the entropy impurity (EI)
            for(int i = 0; i < entropyImpurity.length; i++) {
                entropyImpurity[i] = computeEntropyImpurity(data, quantifyValues, i);
            }

            double minImpurity = Double.POSITIVE_INFINITY;
            double minImpurityValue = Double.NEGATIVE_INFINITY;
            double minImpuritySplitDistribution = Double.NEGATIVE_INFINITY;
            int minImpurityFeature = Integer.MIN_VALUE;

            /**
             * compares calculated EI for each feature
             * and selects the feature with composition of minimal EI and split distribution
             */
            for(int i = 0; i < entropyImpurity.length; i++) {
                if(minImpurity >= entropyImpurity[i][0] && minImpuritySplitDistribution < entropyImpurity[i][3]) {
                    minImpurity = entropyImpurity[i][0];
                    minImpurityValue = entropyImpurity[i][1];
                    minImpurityFeature = (int) entropyImpurity[i][2];
                    minImpuritySplitDistribution = entropyImpurity[i][3];
                }
            }

            // saves the used features
            usedFeature.add(minImpurityFeature);

            node.setDecisionValueBound(minImpurityValue);
            node.setDecisionAttribute(minImpurityFeature);
            deep++;

            // splits the data dependent on the min EI feature for the new nodes
            double[][][] newData = splitData(data, minImpurityFeature, minImpurityValue);

            // builds the children nodes
            node.left = build(newData[0], node, deep);
            node.right = build(newData[1], node, deep);
        }
        return node;
    }

    /**
     * Method calculates the sub-labels for calculating entropy impurity
     * @param data label + feature-set
     * @param featureNumber selected feature
     * @param values
     * @return bound values
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
            //allocates instances in bins
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j];

            }
        }
        return subLabels;
    }


    /**
     * Method calculates the minimal entropy impurity of a selected feature
     * @param data label + feature-set
     * @param quantifyValues bound values
     * @param featureNumber selected feature
     * @return entropy impurity values of the selected feature
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
     * Methods splits the data dependet on a selected feature in bins
     * @param data  label + feature-set
     * @param featureNumber selected feature
     * @param value bound values
     * @return array with data-sets
     */
    private double[][][] splitData(double[][] data, int featureNumber, double value) {

        //sorts the data dependent on a selected feature
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[2][][];
        int distribution;
        distribution = countHitValue(data[featureNumber], Double.NEGATIVE_INFINITY, value);

        splitData[0] = new double[data.length][distribution];
        splitData[1] = new double[data.length][data[0].length-distribution];

        //splits the data
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
     * Method classifies a feature-set
     * @param patterns feature-set
     * @return classified labels
     */
    public double[] classify(double[][] patterns) {
        // calculates the normalised features
        patterns = normalisation(patterns);

        double[] labels = new double[patterns.length];

        // passes each instance throw the tree
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }

    /**
     * Method passes  a given instance of a feature-set throw the tree and returns the label of the leaf
     * @param pattern instance of the feature-set
     * @return classified label
     */
    public double passTree(double[] pattern) {
        BinarySplitNode binarySplitNode = root;
        double classified;

        int feature;
        double value;

        // passes the tree, while node is not a leaf
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

    /**
     * Method saves the tree
     * @param path storage path
     */
    public void saveData(String path) {
        try {
            FileWriter fw = new FileWriter(path);
            save(root, fw);
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method saves the nodes by passing the tree recursively
     * @param node saved node
     * @param fw FileWriter for saving
     */
    private void save(BinarySplitNode node, FileWriter fw) {
        try {
            //if node not a leaf, than save the nodes values
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

            //if node a leaf, than save the leaf values
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

    /**
     * Method loads a saved tree
     * @param path storage path
     */
    public void loadData(String path) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(path));
            ArrayList<String> data = new ArrayList<>();
            String line;

            //read all lines in the file
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

    /**
     * Method builds a decision tree with saved values recursively
     * @param parent parent node
     * @param data data of the node
     * @return built node
     */
    private BinarySplitNode buildWithLoadedData(BinarySplitNode parent, ArrayList<String> data) {

        count++;
        BinarySplitNode node = new BinarySplitNode();
        node.parent = parent;
        String[] parts = data.get(count).split(",");
        //if node is not a leaf, than load the node values
        if(parts[0].equals("false")) {
            node.setLeaf(false);
            node.setDecisionAttribute(Integer.parseInt(parts[1]));
            node.setDecisionValueBound(Double.parseDouble(parts[3]));
            node.left = buildWithLoadedData(node, data);
            node.right = buildWithLoadedData(node, data);
        }
        //if node a leaf, than load the leaf values
        else {
            node.setLeaf(true);
            node.setClassLabel(Double.parseDouble(parts[1]));
            node.left = null;
            node.right = null;
        }
        return node;
    }
}
