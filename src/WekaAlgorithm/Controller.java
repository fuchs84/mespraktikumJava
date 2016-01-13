package WekaAlgorithm;

import WekaAlgorithm.Classifier.*;
import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */

/**
 {{"AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"
 , "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"},
 {"DecisionTree", "DecisionTree","DecisionTree","DecisionTree","DecisionTree","DecisionTree","DecisionTree",
 "DecisionTree","DecisionTree","DecisionTree","DecisionTree"},
 {"RandomForest","RandomForest","RandomForest","RandomForest", "RandomForest","RandomForest", "RandomForest",
 "RandomForest","RandomForest","RandomForest","RandomForest",},
 {"NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes","NaiveBayes",
 "NaiveBayes","NaiveBayes","NaiveBayes"},
 {"SMO","SMO","SMO","SMO","SMO","SMO","SMO","SMO",
 "SMO","SMO","SMO"}};
 */
public class Controller {
    private DataGenerator dataGenerator;
    private ArrayList<AbstractClassifier> classifiers;
    private int numberOfClassifiers;
    private boolean built = false;
    private String[] selectedClassifiers = {"AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"
            , "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost", "AdaBoost"};

    private String path = "evaluationSMO.txt";

    Instances[] allInstances;
    Instances[] passInstances;

    public Controller() {
        dataGenerator = new DataGenerator();
    }

    public void init(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        try {
            passInstances = dataGenerator.buildTrain(patternPathPass, labelPathPass);
            allInstances = dataGenerator.buildTrain(patternPathAll, labelPathAll);
            buildStructure();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    private void buildStructure() throws Exception {
        numberOfClassifiers = allInstances.length + passInstances.length;
        classifiers = new ArrayList<>();
        if(selectedClassifiers.length == numberOfClassifiers) {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier;
                if(selectedClassifiers[i].equals("DecisionTree")) {
                    System.out.println("DecisionTree");
                    classifier = new ClassifierJ48();
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("NaiveBayes")) {
                    System.out.println("NaiveBayes");
                    classifier = new ClassifierNB();
                    classifiers.add(classifier);
                } else if(selectedClassifiers[i].equals("RandomForest")) {
                    System.out.println("RandomForest");
                    classifier = new ClassifierRF();
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("MultilayerPerceptron")) {
                    System.out.println("MultilayerPerceptron");
                    classifier = new ClassifierMLP();
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("AdaBoost")) {
                    System.out.println("AdaBoost");
                    classifier = new ClassifierAB();
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("SMO")) {
                    System.out.println("SMO");
                    classifier = new ClassifierSMO();
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("NBTree")) {
                    System.out.println("NBTree");
                    classifier = new ClassifierNBT();
                    classifiers.add(classifier);
                } else if (selectedClassifiers[i].equals("KNN")) {
                    System.out.println("KNN");
                    classifier = new ClassifierNBT();
                    classifiers.add(classifier);
                }
            }
            built = false;
        } else {
            System.out.println("numberOfClassifiers is not equal to selectedClassifiers");
        }
    }

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
                        eval.crossValidation(classifier.getClassifier(), 10, passInstances[i], stringBuilder);
                    } else {
                        eval.crossValidation(classifier.getClassifier(), 10, allInstances[i - passInstances.length], stringBuilder);
                    }
                    stringBuilder.append("\n" + "\n");
                }
            } else if(type.equals("percentageSplit")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = classifiers.get(i);
                    stringBuilder.append("Fehlerlabel: " + (i+1) + "\n");
                    stringBuilder.append(selectedClassifiers[i] + "\n");
                    if(i < passInstances.length) {
                        eval.percentageSplit(classifier.getClassifier(), passInstances[i], stringBuilder);
                    } else {
                        eval.percentageSplit(classifier.getClassifier(), allInstances[i - passInstances.length], stringBuilder);
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
}
