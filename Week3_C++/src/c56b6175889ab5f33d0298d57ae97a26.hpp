//
//  miniboid.hpp
//  mySketch-week3-1
//
//  
//

#ifndef miniboid_hpp
#define miniboid_hpp

#include <stdio.h>
#include "boid.h"


class miniboid : public Boid {
public:
    miniboid();
    void draw();
};

#endif /* miniboid_hpp */
