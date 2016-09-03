package com.rc;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTM {

	private static Logger log = LoggerFactory.getLogger(LSTM.class);

	int numInputs ;
	int numOutputs ;

	protected MultiLayerNetwork model ;

	public LSTM( int numInputs ) {
		this.numInputs  = numInputs ;
		this.numOutputs = numInputs ;
	}
	
	public void train( DataSetIterator iter  ) {
		model.fit( iter ) ;
	}

	
	public void createModelConfig( int ... layerWidth ) {

		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed( 100 )
				.iterations( 1 )
				.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT )
				.learningRate(0.06)
				//.rmsDecay(0.95)
//				.adamVarDecay(0.99 ) 
//				.adamMeanDecay( 0.99 )
				.regularization(true)  // l1,l2 & dropout enabled ?
				.l2(0.001)
//				.dropOut( 0.001 )		// don't do this - it ruins learning
				.weightInit(WeightInit.XAVIER )
				.updater(Updater.ADAGRAD )
				.list()
				;

		int ni = numInputs ;
		int no = layerWidth[0] ;
		for( int i=0 ; i<layerWidth.length ; i++ ) {
			lb.layer(i, new GravesLSTM.Builder()
					.nIn(ni)
					.nOut(no)
					.activation("softsign")
					.build()
					) ;
			ni = no ;
		}

		lb.layer( layerWidth.length, new RnnOutputLayer.Builder()
				.activation("softmax")
				.lossFunction(LossFunctions.LossFunction.MCXENT )
				.weightInit(WeightInit.XAVIER)
				.nIn(no)
				.nOut(numOutputs)
				.build()
				) ;

		MultiLayerConfiguration conf = lb
				.backprop(true)
//				.backpropType(BackpropType.TruncatedBPTT)
//				.tBPTTForwardLength(100)
//				.tBPTTBackwardLength(100)
				.pretrain(false)
				.build();

		log.debug("Creating new model" ) ;
		model = new MultiLayerNetwork( conf ) ;
		model.init();
		model.setListeners(new ScoreIterationListener(1));
	}
}

