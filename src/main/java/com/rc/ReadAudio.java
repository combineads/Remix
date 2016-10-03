package com.rc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FilterOutputStream;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ReadAudio {
	private static Logger log = LoggerFactory.getLogger(ReadAudio.class);

	AudioFormat sourceFormat  ;
	final AudioFormat targetFormat  ;
	AudioFormat intermediateFormat ;
	
	public ReadAudio() {
	
		targetFormat = new AudioFormat(
				AudioFormat.Encoding.PCM_SIGNED,  // basic raw waveform 
				1_000,  // sample rate
				8, // 8 bits
				1, // channels 
				1, // 8 bit mono = frame size = 1 byte 
				2_000,  // frames/sec = sampling freq @ 1byte/frame
				true 
				);
	}
	
	public byte[] loadMp3( String fileName ) throws Exception {
		log.info( "Loading {}", fileName ) ;
		
		File mp3File = new File( fileName ) ;
		if( !mp3File.canRead() ) {
			throw new Error( "Cannot read input MP3 " + fileName ) ;
		}
		try( AudioInputStream ais = AudioSystem.getAudioInputStream(mp3File) ) {
			sourceFormat = ais.getFormat();

			intermediateFormat = new AudioFormat(
					AudioFormat.Encoding.PCM_SIGNED, 
					sourceFormat.getSampleRate(), 
					16, 
					sourceFormat.getChannels(), 
					sourceFormat.getChannels()*2,   // x2 = bytes per channel ( 16 = 2 bytes ) 
					sourceFormat.getSampleRate(), 
					true 
					);

			AudioInputStream intermediateAis = AudioSystem.getAudioInputStream(intermediateFormat, ais);
			AudioInputStream convertAis = AudioSystem.getAudioInputStream(targetFormat, intermediateAis);
			ByteArrayOutputStream baos = new ByteArrayOutputStream() ;
			byte [] buffer = new byte[8192];
			boolean foundStart = false ;
			while(true){
				int readCount = convertAis.read(buffer, 0, buffer.length);
				if(readCount == -1){
					break;
				}
				if( !foundStart ) {
					for( int i=0 ; i<readCount ; i++ ) {
						if( buffer[i] != 0 ) {
							foundStart = true ;
							break ;
						}
					}					
				}
				if( foundStart ) {
					baos.write(buffer, 0, readCount);
				}
			}
			
			return baos.toByteArray();			
		}		
	}

	
	public void saveWav( String fileName, byte[] data ) throws Exception {
		log.info( "Saving {}", fileName ) ;
		File mp3File = new File( fileName ) ;
		
		try( ByteArrayInputStream bis = new ByteArrayInputStream(data) ;
				AudioInputStream ais = new AudioInputStream(bis, targetFormat, data.length ) ;
				) {
			
			AudioSystem.write(
					ais 
                   ,AudioFileFormat.Type.WAVE
                   ,mp3File ) ;			
		}		
	}
	
	public void saveRaw( String fileName, byte[] data ) throws Exception {
		log.info( "Saving {}", fileName ) ;
		
		try( FileOutputStream fos = new FileOutputStream( fileName ) ) {
			fos.write( data ) ;
			fos.flush();
			log.info( "Saved {} bytes.", data.length ) ;
		}		
		
	}
	
}
