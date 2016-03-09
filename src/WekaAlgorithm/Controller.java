package WekaAlgorithm;

import WekaAlgorithm.Classifier.*;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Controller algorithm
 */
public class Controller {
    private double[] evaluationResults;
    private DataGenerator dataGenerator;
    private ArrayList<AbstractClassifier> classifiers;
    private int numberOfClassifiers;
    private boolean built = false;

    /**
     * Selected classifier (one for each mistake)
     */
    private String[] selectedClassifiers = {"AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"
            , "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"};

    /**
     * Selected options for each classifier
     */
    private String[][] selectedOptions = {  {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""},
                                            {""}};

    /**
     * Storage path for saved results
     */
    private String path = "evaluation.txt";

    /**
     * Instances for the mistakes
     */
    Instances[] allInstances;
    Instances[] passInstances;

    /**
     * Constructor
     */
    public Controller() {
        dataGenerator = new DataGenerator();
    }

    /**
     * Method initialised the Instances
     * @param patternPathAll pattern storage path
     * @param labelPathAll label storage path
     * @param patternPathPass pattern storage path (mistake amble)
     * @param labelPathPass label storage path (mistake amble)
     */
    public void init(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        try {
            passInstances = dataGenerator.buildTrain(patternPathPass, labelPathPass);
            allInstances = dataGenerator.buildTrain(patternPathAll, labelPathAll);
            //buildStructure();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Method builds the structure of the algorithm
     * @throws Exception to invoking method
     */
    public void buildStructure() throws Exception {
        numberOfClassifiers = allInstances.length + passInstances.length;
        evaluationResults = new double[numberOfClassifiers];
        classifiers = new ArrayList<>();
        if(selectedClassifiers.length == numberOfClassifiers) {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier;
                if(selectedClassifiers[i].equals("DecisionTree")) {
                    System.out.println("DecisionTree");
                    classifier = new ClassifierJ48(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("NaiveBayes")) {
                    System.out.println("NaiveBayes");
                    classifier = new ClassifierNB(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("RandomForest")) {
                    System.out.println("RandomForest");
                    classifier = new ClassifierRF(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("MultilayerPerceptron")) {
                    System.out.println("MultilayerPerceptron");
                    classifier = new ClassifierMLP(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("AdaBoost")) {
                    System.out.println("AdaBoost");
                    classifier = new ClassifierAB(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("SMO")) {
                    System.out.println("SMO");
                    classifier = new ClassifierSMO(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("NBTree")) {
                    System.out.println("NBTree");
                    classifier = new ClassifierNBT(selectedOptions[i]);
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("KNN")) {
                    System.out.println("KNN");
                    classifier = new ClassifierNBT(selectedOptions[i]);
                    classifiers.add(classifier);
                }
            }
            built = false;
        } else {
            System.out.println("numberOfClassifiers is not equal to selectedClassifiers");
        }
    }

    /**
     * Method trains all selected classifiers
     */
    public void train() {
        try {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier = classifiers.get(i);
                if(i < passInstances.length) {
                    classifier.setInstances(passInstances[i]);
                } else {
                    classifier.setInstances(allInstances[i - passInstances.length]);
                }
                classifier.train();
                classifiers.set(i, classifier);
            }
            built = true;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Method classifies all mistakes
     * @param patternPath storage path for classifying data
     */
    public void classify(String patternPath) {
        double[][] labels = new double[numberOfClassifiers][];
        try {
            Instances instances = dataGenerator.buildClassify(patternPath);
            if(built == false) {
                train();
            }
            for(int i = 0; i < numberOfClassifiers; i++) {

                AbstractClassifier classifier = classifiers.get(i);
                classifier.setInstances(instances);
                labels[i] = classifier.classify();
            }
            dataGenerator.saveResults(labels);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Method evaluates all selected classifiers
     * @param type evaluation Type
     */
    public void evaluation(String type) {
        try {
            StringBuilder stringBuilder = new StringBuilder();
            ClassifierEvaluation eval = new ClassifierEvaluation();
            if(type.equals("crossValidation")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = classifiers.get(i);
                    stringBuilder.append("Fehlerlabel: " + (i+1) + "\n");
                    stringBuilder.append(selectedClassifiers[i] + "\n");
                    if(i < passInstances.length) {
                        evaluationResults[i] = eval.crossValidation(classifier.getClassifier(), 10, passInstances[i], stringBuilder);
                    } else {
                        evaluationResults[i] = eval.crossValidation(classifier.getClassifier(), 10, allInstances[i - passInstances.length], stringBuilder);
                    }
                    stringBuilder.append("\n" + "\n");
                }
            } else if(type.equals("percentageSplit")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = classifiers.get(i);
                    stringBuilder.append("Fehlerlabel: " + (i+1) + "\n");
                    stringBuilder.append(selectedClassifiers[i] + "\n");
                    if(i < passInstances.length) {
                        evaluationResults[i] = eval.percentageSplit(classifier.getClassifier(), passInstances[i], stringBuilder);
                    } else {
                        evaluationResults[i] = eval.percentageSplit(classifier.getClassifier(), allInstances[i - passInstances.length], stringBuilder);
                    }
                    stringBuilder.append("\n" + "\n");
                }
            }
            FileWriter fw = new FileWriter(path);
            fw.append(stringBuilder);
            fw.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * setter-Method for selected classifiers
     * @param selectedClassifiers selected classifiers
     */
    public void setSelectedClassifiers(String[] selectedClassifiers) {
        this.selectedClassifiers = selectedClassifiers;
    }

    /**
     * setter-Method for selected options for each classifier
      * @param selectedOptions
     */
    public void setSelectedOptions(String[][] selectedOptions) {
        this.selectedOptions = selectedOptions;
    }

    /**
     * getter-Method for evaluation results
     * @return evaluation results
     */
    public double[] getEvaluationResults() {
        return evaluationResults;
    }
}
