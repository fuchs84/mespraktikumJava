package MLP;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Multilayer-Perceptron classifier (own implementation)
 */

public class MLP {
    /**
     * Number of input, output und hidden layers
     */
    private int nInput, nOutput;
    private int[] nHidden;

    /**
     * Weights between the layers
     */
    private List<double[][]> weights = new LinkedList<double[][]>();

    /**
     * Current Values of the input, output und hidden layers
     */
    private double [] input, output;
    private List<double[]> layers = new LinkedList<double[]>();

    /**
     * Number of hidden layers
     */
    private int numberOfHiddenlayers = 1;

    /**
     * Current and desired values of the output
     */
    private double[][] currentOutput;
    private double[][] desiredOutput;

    /**
     * Entire Error
     */
    private double  entireError = 0.0;

    /**
     * Desired output distribution
     */
    private List<Double> desiredOutputDistribution = new LinkedList<Double>();

    /**
     * Method trains the classifier and iterates as long as the entire error decreases.
     * @param patterns Train features
     * @param labels Train labels
     * @param nHidden Number of the perceptrons in the hidden Layers
     * @param learningRate learning Rate (between 0 and 1)
     * @param maxIteration Number of max iterations.
     */
    public void train(double[][] patterns, double[] labels, int[] nHidden, double learningRate, long maxIteration) {
        nInput = patterns[0].length;
        nOutput = computeMaxLabel(labels);
        this.nHidden = nHidden;

        initLayers();
        initWeights();

        patterns = normalisation(patterns);
        double[][] extendedLabels = extendedLabels(labels);


        calculateDistribution(extendedLabels);
        double previousEntireError = Double.POSITIVE_INFINITY;
        int index = 0;
        if (patterns.length == extendedLabels.length) {
            do {
                if (index > 0) {
                    previousEntireError = entireError;
                }
                currentOutput = new double[extendedLabels.length][extendedLabels[0].length];
                desiredOutput = extendedLabels;

                for (int i = 0; i < patterns.length; i++) {

                    currentOutput[i] = passNetwork(patterns[i]);
                    backPropagation(extendedLabels[i], learningRate);
                }

                entireError = 0.0;
                calculateEntireError();
                index++;
            } while (previousEntireError >= entireError && index < maxIteration) ;
            System.out.println("Iterations: " + index);
        } else{
            System.out.println("Pattern und Labels passen nicht zusammen");
        }
    }

    /**
     * Method initialize the all layers
     */
    private void initLayers(){
        numberOfHiddenlayers = nHidden.length;

        input = new double[nInput + 1];
        output = new double[nOutput];

        for (int i = 0; i < numberOfHiddenlayers; i++) {
            double [] hidden = new double[nHidden[i]+1];
            layers.add(hidden);
        }
    }

    /**
     * Method initializes the weights between the layers with random numbers between -0.5 and 0.5
     */
    private void initWeights(){
        for (int i = 0; i <= numberOfHiddenlayers; i++) {
            double[][] weightMatrix;
            if(i== 0) {
                weightMatrix = new double[nInput+1][nHidden[i]+1];
            } else if (i == numberOfHiddenlayers) {
                weightMatrix = new double[nHidden[i-1]+1][nOutput];
            } else {
                weightMatrix = new double[nHidden[i-1]+1][nHidden[i]+1];
            }
            for (int j = 0; j < weightMatrix.length; j++) {
                for (int k = 0; k < weightMatrix[0].length; k++) {
                    weightMatrix[j][k] = Math.random() - 0.5;
                }
            }
            weights.add(weightMatrix);
        }
    }

    /**
     * Method prints the current values of the weights
     */
    public void printWeights() {
        for (int i = 0; i <= numberOfHiddenlayers; i++) {
            double [][] weightMatrix = weights.get(i);
            System.out.println("weight Matrix " + i + ":");
            for (int j = 0; j < weightMatrix.length; j++) {
                for (int k = 0; k < weightMatrix[0].length; k++) {
                    System.out.print(weightMatrix[j][k] + " ");
                }
                System.out.println(" ");
            }
            System.out.println(" ");
        }
    }

    /**
     * Method prints the current values of the layers
     */
    public void printLayers() {
        System.out.println("Inputlayer: ");
        for (int i=0; i < input.length; i++) {
            System.out.print(input[i] + " ");
        }
        System.out.println(" ");
        for (int i = 0; i < numberOfHiddenlayers; i++) {
            System.out.println("Hiddenlayer " + i + ":");
            double[] temp = layers.get(i);
            for (int j=0; j < temp.length; j++) {
                System.out.print(temp[j] + " ");
            }
            System.out.println(" ");
        }
        System.out.println("Outputlayer: ");
        for (int i = 0; i < output.length; i++) {
            System.out.print(output[i] + " ");
        }
    }

    /**
     * Method calculates after each iteration the entire error
     */
    private void calculateEntireError() {
        for(int i = 0; i < currentOutput.length; i++) {
            for(int j = 0; j < currentOutput[0].length; j++) {
                entireError = entireError + 0.5 * Math.pow(desiredOutput[i][j] - currentOutput[i][j], 2);
            }
        }
    }

    /**
     * Method search after the max label in the label-set
     * @param labels label-set
     * @return max label
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
     * Method extends the labels for the output layer. Each label has his own output perceptron.
     * @param labels label-set
     * @return extend label-set
     */
    private double[][] extendedLabels(double[] labels) {
        int length = computeMaxLabel(labels), value;
        double[][] extendedLabels = new double[labels.length][length];

        for (int i = 0; i < labels.length; i ++) {
            value = (int) labels[i];
            extendedLabels[i][value-1] = 1.0;
        }
        return extendedLabels;
    }

    /**
     * Method calculates the normalisation of the features
     * @param patterns feature-set
     * @return normalized feature-set
     */
    private double[][] normalisation(double[][] patterns) {
        double[][] newPatterns = new double[patterns.length][patterns[0].length];
        double max, min;
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
            for(int j = 0; j < patterns.length; j++) {
                newPatterns[j][i] = (patterns[j][i]-min)/(max-min);
            }
        }
        return newPatterns;
    }

    /**
     * Method classifies a feature-set
     * @param patterns feature-set
     * @return classified label-set
     */
    public double[] classify(double[][] patterns) {
        patterns = normalisation(patterns);
        double [] labels = new double[patterns.length];


        for(int i = 0; i < patterns.length; i++) {
            double[] output = passNetwork(patterns[i]);
            labels [i] = (double) winner(output);
        }
        return labels;
    }


    /**
     * Method calculates the strongest output layer. This layer determines the output label
     * @param classified classified output layer
     * @return classified output label
     */
    private int winner(double[] classified) {
        double max = Double.NEGATIVE_INFINITY;
        int index = 0;
        for(int i = 0; i < classified.length; i++) {
            if (classified[i] > max) {
                max = classified[i];
                index = i+1;
            }
        }
        return index;
    }

    /**
     * Method passes a feature-set instance the network.
     * @param trainInput feature-set instance
     * @return activation of the output layer
     */
    private double[] passNetwork(double[] trainInput) {

        input[0] = 1.0;
        for(int i=0; i<nInput; i++) {
            input[i+1] = trainInput[i];
        }

        double[] startLayer = input;
        double[][] weightMatrix = weights.get(0);
        double[] targetLayer = layers.get(0);

        targetLayer[0] = 1.0;
        for (int i=1; i<targetLayer.length; i++) {
            targetLayer[i] = 0.0;
            for(int j=0; j<startLayer.length; j++) {
                targetLayer[i] += weightMatrix[j][i] *startLayer[j];
            }
            targetLayer[i] = activationFunction(targetLayer[i]);
        }
        layers.set(0, targetLayer);

        for (int h=1; h < numberOfHiddenlayers; h++) {
            startLayer = layers.get(h - 1);
            weightMatrix = weights.get(h);
            targetLayer = layers.get(h);

            targetLayer[0] = 1.0;
            for (int i=1; i<targetLayer.length; i++) {
                targetLayer[i] = 0.0;
                for(int j=0; j<startLayer.length; j++) {
                    targetLayer[i] += weightMatrix[j][i] *startLayer[j];
                }
                targetLayer[i] = activationFunction(targetLayer[i]);
            }
            layers.set(h, targetLayer);
        }

        startLayer = layers.get(numberOfHiddenlayers -1);
        weightMatrix = weights.get(numberOfHiddenlayers);
        targetLayer = output;
        for (int i=0; i<targetLayer.length; i++) {
            targetLayer[i] = 0.0;
            for(int j=0; j<startLayer.length; j++) {
                targetLayer[i] = targetLayer[i] +  weightMatrix[j][i] *startLayer[j];
            }
            targetLayer[i] = activationFunction(targetLayer[i]);
        }
        output = targetLayer;
        return output;
    }

    /**
     * Activation function calculates the output value for each perceptron
     * @param value input value
     * @return output value
     */
    private double activationFunction(double value) {
        double solution = 0.0;
        solution = 1.0/(1.0 + Math.exp(-value));
        return solution;
    }

    /**
     * Method calculates the distribution in the train-label-set
     * @param desiredOutput Desired output
     */
    private void calculateDistribution(double [][] desiredOutput) {
        double temp = 0.0;

        int numberOfOutputs = desiredOutput.length;
        for (int i = 0; i < nOutput; i++) {
            desiredOutputDistribution.add(temp);
        }

        for (int i = 0; i < numberOfOutputs; i++) {
            for (int j = 0; j < nOutput; j++) {
                if (desiredOutput[i][j] == 1.0) {
                    temp = desiredOutputDistribution.get(j);
                    temp = temp + 1.0;
                    desiredOutputDistribution.set(j, temp);
                }
            }
        }
        for (int i = 0; i < nOutput; i++) {
            temp = desiredOutputDistribution.get(i);
            temp = temp/((double)numberOfOutputs);
            desiredOutputDistribution.set(i,temp);
        }
    }

    /**
     * Method propagate the output error backwards to the input layer
     * @param desiredOutput calculated output
     * @param learningRate learning rate
     */
    private void backPropagation(double[] desiredOutput, double learningRate) {

        double distributionFactor = 0.0;

        for(int i = 0; i < nOutput; i++) {
            if (desiredOutput[i] == 1.0) {
                distributionFactor =1/(desiredOutputDistribution.get(i)*100.0);
            }
        }

        List<double[]> errors = new LinkedList<double[]>();
        for(int i = 0; i <= numberOfHiddenlayers; i++) {
            double [] error;
            if (i < numberOfHiddenlayers) {
                error = new double[nHidden[i]+1];
            } else {
                error = new double[nOutput];
            }
            errors.add(error);
        }

        double[] error;
        double[][] weight;
        double[][] nextWeight;
        double[] nextError;
        double[] layer;
        double nCurrentLayer, nNextLayer;
        double errorSum = 0.0;



        error = errors.get(numberOfHiddenlayers);
        for (int j = 0; j < nOutput; j++) {
            error[j] = output[j] * (1.0-output[j]) * (desiredOutput[j] - output[j]);
        }
        errors.set(numberOfHiddenlayers, error);


        error = errors.get(numberOfHiddenlayers -1);
        nextError = errors.get(numberOfHiddenlayers);
        nextWeight = weights.get(numberOfHiddenlayers);
        nCurrentLayer = nHidden[numberOfHiddenlayers -1];
        nNextLayer = nOutput;
        layer = layers.get(numberOfHiddenlayers -1);

        for (int i = 0; i< nCurrentLayer; i++) {
            for (int j = 0; j< nNextLayer; j++) {
                errorSum = errorSum + nextWeight[i][j] * nextError[j];
            }
            error[i] = layer[i]*(1.0-layer[i])*errorSum;
            errorSum = 0.0;
        }
        errors.set(numberOfHiddenlayers -1, error);


        for (int i = numberOfHiddenlayers -1; i > 0; i--) {
            error = errors.get(i-1);
            nextError = errors.get(i);
            nextWeight = weights.get(i);
            nCurrentLayer = nHidden[i-1];
            layer = layers.get(i-1);
            nNextLayer = nHidden[i];
            for (int j = 0; j < nCurrentLayer; j++) {
                for (int k = 0; k < nNextLayer; k++) {
                    errorSum = errorSum + nextWeight[j][k] * nextError[k];
                }
                error[j] = layer[j]*(1.0 - layer[j])*errorSum;
                errorSum = 0.0;
            }
            errors.set(i-1, error);
        }

        weight = weights.get(0);
        layer = input;
        error = errors.get(0);

        for(int i = 0; i <= nInput; i++) {
            for(int j = 1; j <= nHidden[0]; j++) {
                weight[i][j] = weight[i][j] + learningRate*error[j]*layer[i]*distributionFactor;
            }
        }
        weights.set(0, weight);


        for(int i = 0; i < numberOfHiddenlayers -1; i++) {
            weight = weights.get(i+1);
            error = errors.get(i+1);
            layer = layers.get(i);
            for(int j = 0; j <=nHidden[i]; j++) {
                for(int k = 1; k <= nHidden[i+1]; k++) {
                    weight[j][k] = weight[j][k] + learningRate*error[k]*layer[j]*distributionFactor;
                }
            }
            weights.set(i+1, weight);
        }

        weight = weights.get(numberOfHiddenlayers);
        layer = layers.get(numberOfHiddenlayers -1);
        error = errors.get(numberOfHiddenlayers);

        for (int i = 0; i <= nHidden[numberOfHiddenlayers -1]; i++) {
            for (int j = 0; j < nOutput; j++) {
                weight[i][j] = weight[i][j] + learningRate*error[j]*layer[i]*distributionFactor;
            }
        }
        weights.set(numberOfHiddenlayers, weight);
    }

    /**
     * Method saves the Multilayer-perceptron data such as weights, number of layers and perceptrons in a CSV-file.
     * You can use it to save the train-results.
     */
    public void saveData() {
        try {
            FileWriter fw = new FileWriter("mlp.csv");
            fw.append(Integer.toString(nInput));
            fw.append(",");
            for(int i = 0; i < nHidden.length; i++) {
                fw.append(Integer.toString(nHidden[i]));
                fw.append(",");
            }
            fw.append(Integer.toString(nOutput));
            fw.append("\n");
            fw.append("\n");

            double[][] weight;
            for(int i = 0; i < weights.size(); i++) {
                weight = weights.get(i);
                for(int j = 0; j < weight.length; j++) {
                    for(int k = 0; k < weight[0].length; k++) {
                        fw.append(Double.toString(weight[j][k]));
                        fw.append(",");
                    }
                    fw.append("\n");
                }
                fw.append("\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method reads the train-results from a CSV-file.
     * You can classify with the saved train-results.
     */
    public void loadData() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("mlp.csv"));
            ArrayList<String> data = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                data.add(line);
            }
            br.close();
            buildWithLoadedData(data);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Method builds the structure of the multilayer-perceptron with the read CSV-data.
     * @param data CSV-data
     */
    private void buildWithLoadedData(ArrayList<String> data) {
        String[] parts = data.get(0).split(",");
        nHidden = new int[parts.length-2];
        for(int i = 0; i < parts.length; i++) {
            if(i == 0) {
                nInput = Integer.parseInt(parts[i]);
            } else if (i == parts.length-1) {
                nOutput = Integer.parseInt(parts[i]);
            } else {
                nHidden[i-1] = Integer.parseInt(parts[i]);
            }
        }
        initLayers();
        initWeights(data);
    }

    /**
     * Method initialized the weights with the read CSV-data.
     * @param data CSV-data
     */
    private void initWeights(ArrayList<String> data){
        String[] parts;
        for (int i = 0; i <= numberOfHiddenlayers; i++) {
            double[][] weightMatrix;
            if(i== 0) {
                weightMatrix = new double[nInput+1][nHidden[i]+1];
                for (int j = 0; j < weightMatrix.length; j++) {
                    parts = data.get(j +2).split(",");
                    for (int k = 0; k < weightMatrix[0].length; k++) {
                        weightMatrix[j][k] = Double.parseDouble(parts[k]);
                    }
                }
            } else if (i == numberOfHiddenlayers) {
                weightMatrix = new double[nHidden[i-1]+1][nOutput];
                for (int j = 0; j < weightMatrix.length; j++) {
                    parts = data.get(j + 2 + nInput + 2 + (i-1) * (nHidden[i-1] + 2)).split(",");
                    for (int k = 0; k < weightMatrix[0].length; k++) {
                        weightMatrix[j][k] = Double.parseDouble(parts[k]);
                    }
                }
            } else {
                weightMatrix = new double[nHidden[i-1]+1][nHidden[i]+1];
                for (int j = 0; j < weightMatrix.length; j++) {
                    parts = data.get(j + 2 + nInput + 2 + (i-1) * (nHidden[i-1] + 2)).split(",");
                    for (int k = 0; k < weightMatrix[0].length; k++) {
                        weightMatrix[j][k] = Double.parseDouble(parts[k]);
                    }
                }
            }
            weights.add(weightMatrix);
        }
    }
}
