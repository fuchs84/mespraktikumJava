package DT.LMDT;

import DT.DecisionTree;

import java.util.Collections;

/**
 * Created by MatthiasFuchs on 26.11.15.
 */
public class LinearMachineDT extends DecisionTree{

    private LinearMachineNode[] linearMachineNodes;
    private int numberOfClasses;
    private int numberOfFeatures;


    /**
     * Methode trainiert den Decision Tree
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     */
    public void train(double[][] patterns, double[] labels){
        patterns = extendPatterns(patterns);
        searchUsedLabels(labels);
        Collections.sort(usedLabels);
        numberOfClasses = usedLabels.size();
        numberOfFeatures = patterns[0].length;
        numberOfInstances = patterns.length;
        entireDistribution = computeClassDistribution(labels);
        initNodes();
        learn(patterns, labels);
    }

    /**
     * Methode passt die einzelnen Gewichte der Knoten an
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     */
    private void learn(double[][] patterns, double[] labels) {
        double B = 2.0;
        double a = 0.99, b = 0.00005;
        double[] instance;
        double labelClass;
        double[] weightI, weightJ;
        double correction;
        int random;
        int nodeNumber;
        while (B > 0.001 && checkNodes(patterns, labels) < 0.99) {

            random = random();

            instance = patterns[random];
            labelClass = labels[random];
            if(passTree(instance) != labelClass && patterns[random][patterns[0].length-1] == 1.0) {
                System.out.println(checkNodes(patterns, labels));
                nodeNumber = searchNodeNumber(labelClass);
                weightI = linearMachineNodes[nodeNumber].getWeights();
                for(int j = 0; j < numberOfClasses; j++) {
                    if (j != nodeNumber) {
                        weightJ = linearMachineNodes[j].getWeights();
                        correction = computeC(B, weightI, weightJ, instance);
                        for(int k = 0; k < weightI.length; k++) {
                            weightI[k] = weightI[k] + instance[k]*correction;
                            weightJ[k] = weightJ[k] - instance[k]*correction;
                        }
                        linearMachineNodes[j].setWeights(weightJ);

                    }
                }
                linearMachineNodes[nodeNumber].setWeights(weightI);
                B = a*B-b;

                patterns[random][patterns[0].length-1] = 0.0;
            }
        }
    }

    /**
     * Methode berechnet den Korrekturfaktor
     * @param B Train-Konstante
     * @param weightI Gewichte des I-Knotens
     * @param weightJ Gewichte des J-Knotens
     * @param instance Instanz nach der korrigiert wird
     * @return Korrekturfaktor
     */
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

    /**
     * Methode ueberprueft die Knoten wie gut die Gewichte angepasst sind
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @return prozentualer Anteil an richtig klassifizierten Labels
     */
    private double checkNodes(double[][] patterns, double[] labels) {
        int numberOfWrongs = 0, numberOfRights;
        double percent = 0.0;
        double nodeDecision, neighbourDecision;
        int nodeNumber = -1;
        for(int i = 0; i < patterns.length; i++) {
            nodeNumber = searchNodeNumber(labels[i]);
            nodeDecision = computeDecision(patterns[i], nodeNumber);
            for(int j = 0; j < numberOfClasses; j++) {
                if(j != nodeNumber) {
                    neighbourDecision = computeDecision(patterns[i], j);
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

    private int searchNodeNumber(double label) {
        for (int i = 0; i < numberOfClasses; i++) {
            if(label == linearMachineNodes[i].getClassLabel()) {
                return i;
            }
        }
        return -1;
    }


    private double computeDecision(double [] instance, int nodeNumber) {
        double decision = 0.0;

        double [] weight = linearMachineNodes[nodeNumber].getWeights();
        for (int i = 0; i < weight.length; i++) {
            decision += weight[i]*instance[i];
        }
        return decision;
    }

    /**
     * Methode erweitert die Patterns um eine Konstante (Bias)
     * @param patterns Patterns
     * @return erweiterte Patterns
     */
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

    /**
     * Methode initialisiert die Knoten
     */
    private void initNodes() {
        linearMachineNodes = new LinearMachineNode[numberOfClasses];
        for (int i = 0; i < linearMachineNodes.length; i++) {
            linearMachineNodes[i] = new LinearMachineNode();
            linearMachineNodes[i].setClassLabel(usedLabels.get(i));
            linearMachineNodes[i].initWeights(numberOfFeatures);
        }
    }

    /**
     * Methode berechnet einen Zufallswert zur Trainingsauswahl
     * @return Zufallswert
     */
    private int random() {
        int random = (int) (Math.random()*(numberOfInstances));
        return random;
    }

    /**
     * Methode geht durch den Baum und liefert die jeweilige Klasse zurueck
     * @param instance klassifizierende Instanz
     * @return klassifiziertes Label
     */
    private double passTree (double[] instance) {
        double classified = 0.0;
        double decision;
        double maxDecision = Double.NEGATIVE_INFINITY;
        for(int i = 0; i < numberOfClasses; i++) {
            decision = computeDecision(instance, i);
            if(maxDecision < decision) {
                maxDecision = decision;
                classified = linearMachineNodes[i].getClassLabel();
            }
        }
        return classified;
    }

    /**
     * Methode klassifiziert die uebergebenen Patterns
     * @param patterns Patterns die klassifiziert Werden
     * @return double-Array mit den jeweiligen Labels
     */
    public double[] classify(double[][] patterns) {
        patterns = extendPatterns(patterns);
        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }
}
