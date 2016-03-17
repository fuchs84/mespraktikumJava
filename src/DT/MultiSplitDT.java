package DT;

import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;

/**
 * Multi-split decision Tree
 */
public class MultiSplitDT extends DecisionTree {

    //Root node
    private MultiSplitNode root;

    //default split bounds for standardised features
    private double[] values = {Double.NEGATIVE_INFINITY, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};

    //counter for data saving
    private int count;

    /**
     * Method trains the classifier
     * @param patterns feature-set
     * @param labels label-set
     * @param maxDeep maximal depth
     * @param minNodeSize minimal numbers of instances in a Node;
     * @param quantifySize number of split parts for decision rules
     */
    public void train (double[][] patterns, double[] labels, int maxDeep, int minNodeSize, int quantifySize) {
        this.quantifySize = quantifySize;
        this.deep = maxDeep;
        this.minNodeSize = minNodeSize;
        double[][] merge;
        double[][] data;

        //calculates the standardised features
        patterns = standardization(patterns);

        //merge label-set and feature set to data and afterwards transpose the data
        merge = merger(patterns, labels);
        data = transpose(merge);

        //calculates the split bounds
        if(quantifySize >= 3) {
            values = new double[quantifySize + 1];
            for (int i = 0; i <= quantifySize; i++) {
                if (i == 0) {
                    values[i] = Double.NEGATIVE_INFINITY;
                } else if (i == 1) {
                    values[1] = -3.0;
                } else if (i == quantifySize - 1) {
                    values[i] = 3.0;
                } else if (i == quantifySize) {
                    values[i] = Double.POSITIVE_INFINITY;
                } else {
                    values[i] = -3.0 + (i-1)*6.0/(double)(quantifySize - 2);
                }
            }
        } else {
            this.quantifySize = 8;
        }

        numberOfInstances = labels.length;

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
    public MultiSplitNode build (double[][] data, MultiSplitNode parent, int deep) {
        // new node
        MultiSplitNode node = new MultiSplitNode();
        node.parent = parent;
        node.deep = deep;

        // builds leaf if maximal depth or minimal number of instances in a node
        if(this.deep <= deep || minNodeSize > data[0].length) {
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
            return node;
        }

        // builds leaf if the instances of a node are pure
        else if(isNodePure(data[data.length-1])) {
            node.setLeaf(true);
            node.setClassLabel(data[data.length - 1][0]);
            node.children = null;
            return node;
        }

        // builds node with decision rule
        else {
            node.setLeaf(false);
            double maxIG = Double.NEGATIVE_INFINITY;
            int maxIGFeature = Integer.MIN_VALUE;
            double informationGain;

            // compares calculated information gain (IG) for each feature and selects the feature with maximal IG
            for(int i = 0; i < data.length -1; i++) {
                informationGain = computeInformationGain(data, i);
                if(informationGain > maxIG /*&& !usedFeature.contains(i)*/) {
                    maxIG = informationGain;
                    maxIGFeature = i;
                }
            }

            // saves the used features
            usedFeature.add(maxIGFeature);


            node.setDecisionAttribute(maxIGFeature);

            // splits the data dependent on the max IG feature for the new nodes
            double[][][] newData = splitData(data, maxIGFeature);

            // search after the data-splits without instances
            int[] distribution = new int[quantifySize];
            int index = 0;
            for(int i = 0; i < quantifySize; i++) {
                distribution[i] = newData[i][0].length;
                if(newData[i][0].length > 0) {
                    index++;
                }
            }

            // new nodes for the data-splits
            node.children = new MultiSplitNode[index];

            // calculates the decision values for the node (if data-split zero than ignore)
            double[] newValues = new double[index+1];
            newValues[0] = Double.NEGATIVE_INFINITY;
            index = 1;
            for(int i = 0; i < quantifySize; i++) {
                if (distribution[i] == 0) {
                    if(i == 0) {
                        newValues[index] = values[2];
                    }
                    else if(i == quantifySize-1) {
                        newValues[newValues.length-1] = values[i+1];
                    }
                } else {
                    newValues[index] = values[i+1];
                    index++;
                }
            }

            node.setDecisionValues(newValues);

            deep++;

            // builds the children nodes
            index = 0;
            for (int i = 0; i < quantifySize; i++) {
                if(newData[i][0].length > 0) {
                    node.children[index] = build(newData[i], node, deep);
                    index++;
                }
            }
            return node;
        }
    }


    /**
     * Method splits the data dependent on a selected feature in bins
     * @param data label + feature-set
     * @param featureNumber selected feature
     * @return array with data-sets
     */
    private double[][][] splitData(double[][] data, int featureNumber) {

        //sorts the data dependent on a selected feature
        double[][] dataSort = sort(data, featureNumber);

        double[][][] splitData = new double[quantifySize][][];
        int[] distribution = new int[quantifySize];
        double upperBound, lowerBound;

        //count the instances for each bin
        for (int i = 0; i < quantifySize; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            distribution[i] = countHitValue(data[featureNumber], lowerBound, upperBound);
        }
        int offset = 0;

        //splits the data
        for (int i = 0; i < quantifySize; i++) {
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
     * Method calculates the information gain for a selected feature
     * @param data label + feature-set
     * @param featureNumber selected feature
     * @return information gain
     */
    private double computeInformationGain(double[][] data, int featureNumber) {
        double[] labels = data[data.length-1];
        double[][] subLabels;
        double probability;
        double gain;
        subLabels = computeSubLabels(data, featureNumber);

        //calculates entropy and afterwards subtracts the probability multiplied with the entropy of the sub-labels
        gain = computeEntropy(labels);
        for (int j = 0; j < quantifySize; j++) {
            probability = ((double)subLabels[j].length)/((double)labels.length);

            gain -= (probability) * computeEntropy(subLabels[j]);
        }
        return gain;
    }

    /**
     * Method calculates the sub-labels for calculating information gain
     * @param data label + feature-set
     * @param featureNumber selected feature
     * @return sub-labels
     */
    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[quantifySize][];
        double upperBound, lowerBound;
        int count, offset = 0;
        for (int i = 0; i < quantifySize; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            //allocates instances in bins
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j+offset];
            }
            offset += count;
        }
        return subLabels;
    }

    /**
     * Method classifies a feature-set
     * @param patterns feature-set
     * @return classified labels
     */
    public double[] classify(double[][] patterns) {

        // calculates the standardised features
        patterns = standardization(patterns);

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
    private double passTree(double[] pattern) {
        MultiSplitNode node = root;
        double classified, upperBound, lowerBound;

        int feature;
        double[] values;
        // passes the tree, while node is not a leaf
        while (node.getLeaf() == false) {
            feature = node.getDecisionAttribute();
            values = node.getDecisionValues();
            for (int i = 0; i < values.length-1; i++) {
                lowerBound = values[i];
                upperBound = values[i + 1];
                //search the right child node
                if (lowerBound <= pattern[feature] && pattern[feature] < upperBound) {
                    node = node.children[i];
                }
            }
        }
        classified = node.getClassLabel();
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
    private void save(MultiSplitNode node, FileWriter fw) {
        try {
            //if node not a leaf, than save the nodes values
            if(node.getLeaf() == false) {
                fw.append(Boolean.toString(false));
                fw.append(",");
                fw.append(Integer.toString(node.getDecisionAttribute()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append(",");
                for(int i = 0; i < node.getDecisionValues().length; i++) {
                    fw.append(Double.toString(node.getDecisionValues()[i]));
                    fw.append(",");
                }
                fw.append("\n");
                for(int i = 0; i < node.children.length; i++) {
                    save(node.children[i], fw);
                }
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
    private MultiSplitNode buildWithLoadedData(MultiSplitNode parent, ArrayList<String> data) {

        count++;
        MultiSplitNode node = new MultiSplitNode();
        node.parent = parent;
        String[] parts = data.get(count).split(",");
        //if node is not a leaf, than load the node values
        if(parts[0].equals("false")) {
            node.setLeaf(false);
            node.setDecisionAttribute(Integer.parseInt(parts[1]));
            int valueLength = parts.length -3;
            double[] decisionValues = new double[valueLength];
            for(int i = 0; i < decisionValues.length; i++) {
                decisionValues[i] = Double.parseDouble(parts[i + 3]);
            }
            node.setDecisionValues(decisionValues);
            node.children = new MultiSplitNode[valueLength-1];
            for(int i = 0; i < node.children.length; i++) {
                node.children[i] = buildWithLoadedData(node, data);
            }
        }
        //if node a leaf, than load the leaf values
        else {
            node.setLeaf(true);
            node.setClassLabel(Double.parseDouble(parts[1]));
            node.children = null;
        }
        return node;
    }
}
