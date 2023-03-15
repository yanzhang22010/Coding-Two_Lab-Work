//
//  miniboid.cpp
//  mySketch-week3-1
//
// 
//

#include "miniboid.hpp"

miniboid::miniboid()
{
    separationWeight = 1.9f;
    cohesionWeight = 1.2f;
    alignmentWeight = 0.2f;
    
    separationThreshold = 20;
    neighbourhoodSize = 100;
    
    position = ofVec3f(ofRandom(0, 200), ofRandom(0, 200));
    velocity = ofVec3f(ofRandom(-2, 2), ofRandom(-2, 2));
}

void miniboid::draw(){
    ofSetColor(255, 255, 0);
    ofCircle(position.x, position.y, 20);
}
