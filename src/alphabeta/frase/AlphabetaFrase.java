/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package alphabeta.frase;

import java.util.Arrays;
import javax.swing.JOptionPane;
import weka.classifiers.lazy.IBk;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author rodrigo
 */
public class AlphabetaFrase {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource arff = new ConverterUtils.DataSource("frases.arff");
        Instances frases = arff.getDataSet();
        frases.setClassIndex(3);
        
        int qtdeExemplos = frases.numInstances();
        int qtdeAtributos = frases.numAttributes();
        double[][] matrizFrases = new double[qtdeExemplos][qtdeAtributos];
        System.out.println("EXEMPLOS DA BASE");
        for (int i = 1; i < qtdeExemplos; i++){
            matrizFrases[i-1] = frases.get(i).toDoubleArray();
            System.out.print("Exemplo (" + i + "): ");
            System.out.println(Arrays.toString(matrizFrases[i - 1]));
        }
        
        IBk nnWeka = new IBk();
        
        // TREINANDO O CLASSIFICADOR
        nnWeka.buildClassifier(frases);
        
        //DEFININDO UM EXEMPLO DE TESTE
        // - Definindo quantidade de  atributos
        Instance exemploTeste = new DenseInstance(4);
        
        // - Definindo como serão esses 4 atributos
        exemploTeste.setDataset(frases);
        
        // - Definindo valores desses atributos
        exemploTeste.setValue(0, Double.parseDouble(JOptionPane.showInputDialog("Digite a quantidade de palavras: ")));//(att, value);
        exemploTeste.setValue(1, Double.parseDouble(JOptionPane.showInputDialog("Digite a quantidade de parâmetros: ")));//(att, value);
        exemploTeste.setValue(2, Double.parseDouble(JOptionPane.showInputDialog("Digite a quantidade de letras da frase: ")));//(att, value);

        //exemploTeste.setValue(0, 21);
        //exemploTeste.setValue(1, 10);
        //exemploTeste.setValue(2, 80);
        
        // ENCONTRE A CLASSE GRUPO PARA O EXEMPLO DEFINIDO
        double classWeka = nnWeka.classifyInstance(exemploTeste);
        System.out.println("Classe Weka: " + classWeka);
        System.out.println("Classe Nominal: " + frases.classAttribute().value((int) classWeka));
        JOptionPane.showMessageDialog(null, "A frase foi classificada como: \nClasse Weka: " + classWeka + "\nClasse nominal: " + frases.classAttribute().value((int) classWeka) + ".");
    }
    
    public static double getEuclidiana(double[] vet1, double[] vet2){
        double dist = 0;
        for (int i = 0; i < vet1.length-1; i++){
            dist += Math.pow(vet1[i] - vet2[i], 2);
        }
        dist = Math.sqrt(dist);
        return dist;
    }
}
