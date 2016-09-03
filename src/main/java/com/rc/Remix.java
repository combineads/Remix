package com.rc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Remix {
	private static Logger log = LoggerFactory.getLogger(Remix.class);

	ReadAudio audioReader = new ReadAudio() ;
	static String dataDir ;
	private static String[] mp3s = new String[] {
			"Selena Gomez - Good For You.mp3"
	} ;

	private Random rng = new Random( 50 ) ;

	public static int BATCH_SIZE = 10;
	public static int HISTORY_LENGTH = 1000;
	public static int NUM_TUNES = mp3s.length ;

	private LSTM nn ;
	private byte[] data ;
	private int maxStart ;	
	private int leadingSilence  ;
	
	public Remix() throws Exception {
		data = audioReader.loadMp3( mp3s[0] ) ;
		log.info( "Read {} bytes of audio", data.length ) ;
		audioReader.saveWav( dataDir + "/SelenaNew.wav", data);
		audioReader.saveRaw( dataDir + "/SelenaNew.bin", data);

		maxStart = data.length - HISTORY_LENGTH - 1 ;
		
		for( int i=1 ; i<data.length ; i++ ) {		
			if( data[i] != 0 ) {
				leadingSilence = i-1 ;
				break ;
			}
		}
		nn = new LSTM( 128 ) ;
		nn.createModelConfig( 400, 300, 200 ) ;		
	}

	private DataSetIterator getDatasetIterator() throws Exception {
		
		log.info( "Building new dataset" ) ;
		List<DataSet> rc = new ArrayList<>() ;

		for( int b=0 ; b<BATCH_SIZE ; b++ ) {
			int start = (int)(rng.nextDouble() * (maxStart-leadingSilence) ) + leadingSilence ;
			
			int ix[] = new int[] { 0, 0, 0 } ;

			INDArray features = Nd4j.create( new int[] { 1, nn.numInputs, HISTORY_LENGTH }, 'f' ) ;
			INDArray labels   = Nd4j.create( new int[] { 1, nn.numOutputs, HISTORY_LENGTH }, 'f' ) ;

			for( ix[2]=0 ; ix[2]<HISTORY_LENGTH ; ix[2]++, start++ ) {
				
				int n = data[start-1] + 64  ;   // PCM signed   
				if( n<0 ) n = 0 ;
				if( n>127 ) n = 127 ;
				ix[1] = n ;
				features.putScalar( ix, 1 ) ;
				
				n = data[start+1] + 64 ;
				if( n<0 ) n = 0 ;
				if( n>127 ) n = 127 ;
				ix[1] = n ;
				labels.putScalar( ix, 1 ) ;
			}
			DataSet ds = new DataSet( features,  labels ) ;
			rc.add( ds ) ;
		}

		return new ListDataSetIterator(rc) ;	
	}


	public byte[] test( int size ) throws Exception {

		nn.model.rnnClearPreviousState();

		INDArray input  = Nd4j.zeros( new int[] { 1, nn.numInputs, 1 }  ) ;
		int init = rng.nextInt(128) ;
		input.putScalar( new int[]{ 0, init, 0}, 1.0 ) ;
		
		INDArray output = nn.model.rnnTimeStep( input ) ;
		output = output.tensorAlongDimension(output.size(2)-1,1,0) ;

		int ix[] = new int[] { 0, 0 } ;
		byte buf[] = new byte[ size ] ;
		for( int i=0 ; i<buf.length ; i++ ) {
			if( (i%10_000) == 0 ) {
				log.info( "Created {} bytes of test data", i ) ;
			}
			input  = Nd4j.zeros( new int[] { 1, nn.numInputs }  ) ;   // allowed to make this a 2D array for a single timestep
						
			double r = rng.nextDouble() ;
			double tot = 0 ;
			for( int c=0 ; c<nn.numInputs; c++ ) {
				tot += output.getDouble( ix[0], c ) ;	
				if( tot > r ) {
					ix[1] = c ;
					input.putScalar( ix, 1.0 ) ;
					if( ix[0]==0 ) {
						buf[i] = (byte)( (c-64) ) ;   // 7bits signed ...
					}
					break ;
				}
			}			
			
			output = nn.model.rnnTimeStep( input ) ;
		}
		return buf ;
	}

	public void train() throws Exception {
		DataSetIterator dsi = getDatasetIterator() ;
		nn.train( dsi ) ;
	}


	public void predict( int size ) throws Exception {		
		byte[] data = test( size ) ;		
		audioReader.saveWav( dataDir + "/SelenaNew.wav", data);
		audioReader.saveRaw( dataDir + "/SelenaNew.bin", data);
	}

	public static void main( String args[] ) {
		dataDir = "/home/richard/Downloads" ;
		if( args.length > 0 ) {
			dataDir = args[0] ;
		}
		for( int i=0 ; i<mp3s.length ; i++ ) {
			mp3s[i] = dataDir + "/" + mp3s[i] ;
		}
		
		try {
			Remix self = new Remix() ;
//			if( 1==1) return ;
			Nd4j.ENFORCE_NUMERICAL_STABILITY = true ;
			for( int i=0 ; i<2000 ; i++ ) {
				self.train() ;
				if( (i % 10 ) == 9 ) {
					self.predict( 50_000 );
				}
			}
			self.predict( 1_000_000 );
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
