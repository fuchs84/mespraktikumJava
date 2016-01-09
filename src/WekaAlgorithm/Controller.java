package WekaAlgorithm;

import WekaAlgorithm.Classifier.AbstractClassifier;
import WekaAlgorithm.Classifier.ClassifierJ48;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 08.01.16.
 */
public class Controller {
    private DataGenerator dataGenerator;
    private ArrayList<AbstractClassifier> classifiers;
    private int numberOfClassifiers;

    Instances[] allInstances;
    Instances[] passInstances;

    public Controller() {
        dataGenerator = new DataGenerator();
    }

    public void init(String patternPathAll, String labelPathAll, String patternPathPass, String labelPathPass) {
        try {
            passInstances = dataGenerator.buildTrain(patternPathPass, labelPathPass);
            allInstances = dataGenerator.buildTrain(patternPathAll, labelPathAll);
            numberOfClassifiers = allInstances.length + passInstances.length;
            classifiers = new ArrayList<>();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void train() {
        try {
            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier = new ClassifierJ48();
                classifier.setMode(false);
                if(i < passInstances.length) {
                    classifier.setInstances(passInstances[i]);
                } else {
                    classifier.setInstances(allInstances[i - passInstances.length]);
                }
                classifiers.add(classifier);
                new Thread(classifier).start();
                System.out.println(i);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void classify(String patternPath) {
        double[][] labels = new double[numberOfClassifiers][];
        try {
            Instances instances = dataGenerator.buildClassify(patternPath);

            for(int i = 0; i < numberOfClassifiers; i++) {
                AbstractClassifier classifier = classifiers.get(i);
                classifier.setInstances(instances);
                labels[i] = classifier.classify();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void evaluation(String type) {
        try {
            ClassifierEvaluation eval = new ClassifierEvaluation();
            if(type.equals("crossvalidation")) {
                for(int i = 0; i < numberOfClassifiers; i++) {
                    AbstractClassifier classifier = new ClassifierJ48();
                    if(i < passInstances.length) {
                        eval.crossValidation(classifier.getClassifier(), 10, passInstances[i]);
                    } else {
                        eval.crossValidation(classifier.getClassifier(), 10, allInstances[i - passInstances.length]);
                    }
                }
            } else if(type.equals("traindata")) {

            } else if(type.equals("testdata")) {

            } else if(type.equals("percentsplit")) {

            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
