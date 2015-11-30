package DT.LMDT;

import DT.DecisionTree;

/**
 * Created by MatthiasFuchs on 26.11.15.
 */
public class LinearMachineDT extends DecisionTree{

    private LinearMachineNode[] linearMachineNodes;
    private int numberOfClasses;
    private int numberOfFeatures;

    public void train(double[][] patterns, double[] labels){
        patterns = extendPatterns(patterns);

        numberOfClasses = computeMaxLabel(labels);
        numberOfFeatures = patterns[0].length;
        numberOfInstances = patterns.length;

        initNodes();
        learn(patterns, labels);
    }

    private void learn(double[][] patterns, double[] labels) {
        double B = 2.0;
        double a = 0.99, b = 0.0005;
        double[] instance;
        double labelClass;
        double[] weightI, weightJ;
        double correction;
        int random;
        while (B > 0.001 && checkNodes(patterns, labels) < 0.99) {

            random = random();

            instance = patterns[random];
            labelClass = labels[random];
            if(passTree(instance) != labelClass && patterns[random][patterns[0].length-1] == 1.0) {
                System.out.println(checkNodes(patterns, labels));
                weightI = linearMachineNodes[(int)labelClass].getWeights();
                for(int j = 0; j <= numberOfClasses; j++) {
                    if (j != (int)labelClass) {
                        weightJ = linearMachineNodes[j].getWeights();
                        correction = computeC(B, weightI, weightJ, instance);
                        for(int k = 0; k < weightI.length; k++) {
                            weightI[k] = weightI[k] + instance[k]*correction;
                            weightJ[k] = weightJ[k] - instance[k]*correction;
                        }
                        linearMachineNodes[j].setWeights(weightJ);

                    }
                }
                linearMachineNodes[(int)labelClass].setWeights(weightI);
                //B = a*B-b;

                patterns[random][patterns[0].length-1] = 0.0;
            }

        }
    }

    private double computeC(double B, double[] weightI, double[] weightJ, double[] instance) {
        double c;
        double k = 0.0;
        double temp = 0.0;
        for(int i = 0; i < weightI.length; i++) {
            k += (weightJ[i]-weightI[i])*instance[i];
            temp += instance[i]*instance[i];
        }
        k = k/(2*temp);
        c = B/(B+k);
        return c;
    }


    private double checkNodes(double[][] patterns, double[] labels) {
        int numberOfWrongs = 0, numberOfRights;
        double percent = 0.0;
        double nodeDecision, neighbourDecision;
        for(int i = 0; i < patterns.length; i++) {
            nodeDecision = computeDecision(patterns[i], labels[i]);
            for(int j = 0; j < linearMachineNodes.length; j++) {
                if((double)j != labels[i]) {
                    neighbourDecision = computeDecision(patterns[i], (double)j);
                } else {
                    break;
                }
                if(nodeDecision < neighbourDecision) {
                    numberOfWrongs++;
                    break;
                }
            }
            numberOfRights = numberOfInstances - numberOfWrongs;
            percent = ((double)numberOfRights)/((double)numberOfInstances);
        }
        return percent;
    }

    private double computeDecision(double [] instance, double label) {
        double decision = 0.0;
        double [] weight = linearMachineNodes[(int)label].getWeights();
        for (int i = 0; i < weight.length; i++) {
            decision += weight[i]*instance[i];
        }
        return decision;
    }

    private double[][] extendPatterns(double[][] patterns) {
        double[][] extendedPatterns = new double[patterns.length][patterns[0].length+1];
        for(int i = 0; i < extendedPatterns.length; i++) {
            for(int j = 0; j < extendedPatterns[0].length; j++) {
                if(j == extendedPatterns[0].length-1) {
                    extendedPatterns[i][j] = 1.0;
                }
                else {
                    extendedPatterns[i][j] = patterns[i][j];
                }
            }
        }
        return extendedPatterns;
    }

    private void initNodes() {
        linearMachineNodes = new LinearMachineNode[numberOfClasses+1];
        for (int i = 0; i < linearMachineNodes.length; i++) {
            linearMachineNodes[i] = new LinearMachineNode();
            linearMachineNodes[i].setClassLabel((double)i);
            linearMachineNodes[i].initWeights(numberOfFeatures);
        }
    }

    private int random() {
        int random = (int) (Math.random()*(numberOfInstances));
        return random;
    }


    private double passTree (double[] instance) {
        double classified = 0.0;
        double decision;
        double maxDecision = Double.NEGATIVE_INFINITY;
        for(int i = 0; i <= numberOfClasses; i++) {
            decision = computeDecision(instance, (double)i);
            if(maxDecision < decision) {
                maxDecision = decision;
                classified = (double)i;
            }
        }
        return classified;
    }

    public double[] classify(double[][] patterns) {
        patterns = extendPatterns(patterns);
        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }
}
