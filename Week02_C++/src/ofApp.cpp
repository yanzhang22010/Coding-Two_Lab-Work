#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup(){
    ofBackground(255, 255, 255);
    ofSetFrameRate(15);
}

//--------------------------------------------------------------
void ofApp::update(){
    if (!paused) {
        myLine.addVertex( ofRandom( ofGetWidth()), ofRandom( ofGetHeight()));
    }
}

//--------------------------------------------------------------
void ofApp::draw(){
    int stepSize = 10;
    ofColor myDrawCol;
    myDrawCol.setHsb(30, 255, 255);
    ofSetColor(myDrawCol);
    ofSetLineWidth(3);
    myLine.draw();
    }
       
    

//--------------------------------------------------------------
void ofApp::keyPressed(int key){
        //if (key == 'f'){
            //ofToggleFullscreen();
        //}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
    paused = true;
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){
    paused = false;
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
