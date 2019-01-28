/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

const double MIN = 0.000001;
const double MAX = 100000.0;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50; //Set number of particles
	weights.resize(num_particles); //resize weights
	particles.resize(num_particles); //resize particles

	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	//Initialize particles
	for(int i=0; i<num_particles; i++){
		particles[i].id = i;
		particles[i].x = N_x(gen);
		particles[i].y = N_y(gen);
		particles[i].theta = N_theta(gen);
		particles[i].weight = 1.0;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for(int i = 0; i < num_particles; i++){

		if(fabs(yaw_rate) < MIN){
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
			//particles[i].theta;
		}
		else{
			particles[i].x += velocity * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) / yaw_rate;
			particles[i].y += velocity * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) / yaw_rate;
			particles[i].theta += yaw_rate * delta_t;
		}

		normal_distribution<double> N_x(0, std_pos[0]);
		normal_distribution<double> N_y(0, std_pos[1]);
		normal_distribution<double> N_theta(0, std_pos[2]);

		//Add noise
		particles[i].x += N_x(gen);
		particles[i].y += N_y(gen);
		particles[i].theta += N_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	unsigned int i, j;
	for(i = 0; i < observations.size(); i++){
		double minDistance = numeric_limits<double>::max();
		int mapId = -1;

		for(j = 0; j < predicted.size(); j++){
			double distance = sqrt(pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2));
			if(distance < minDistance){
				minDistance = distance;
				mapId = predicted[j].id;
    	}
    }
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
	vector<double> landmark_obs_dist (landmarks.size());
	double cos_theta, sin_theta, obs_x, obs_y, new_obs_x, new_obs_y;
	double p = 2 * std_landmark[0] * std_landmark[0];
    double q = 2 * std_landmark[1] * std_landmark[1];
    double r = 2 * M_PI * std_landmark[0] * std_landmark[1];
	unsigned int j, k;

    for(int i = 0; i < num_particles; i++){
        double gauss = 1.0;
        for(j = 0; j < observations.size(); j++){
        	cos_theta = cos(particles[i].theta);
        	sin_theta = sin(particles[i].theta);
        	obs_x = observations[j].x;
        	obs_y = observations[j].y;

        	//Transformed Observation Points
            new_obs_x = obs_x * cos_theta - obs_y * sin_theta + particles[i].x;
            new_obs_y = obs_x * sin_theta + obs_y * cos_theta + particles[i].y;

            //Calculate distance to nearest landmark neighbour
            for(k = 0; k < landmarks.size(); k++){
                double landmark_part_dist = sqrt(pow(particles[i].x - landmarks[k].x_f, 2) + pow(particles[i].y - landmarks[k].y_f, 2));
                
                if(landmark_part_dist <= sensor_range)
                	landmark_obs_dist[k] = sqrt(pow(new_obs_x - landmarks[k].x_f, 2) + pow(new_obs_y - landmarks[k].y_f, 2));
                else
                	landmark_obs_dist[k] = MAX;
            }
            
            //Associate the observation point to nearest landmark neighbour
            int min_pos = distance(landmark_obs_dist.begin(),min_element(landmark_obs_dist.begin(),landmark_obs_dist.end()));
            float nn_x = landmarks[min_pos].x_f;
            float nn_y = landmarks[min_pos].y_f;
            
            double x_diff = new_obs_x - nn_x;
            double y_diff = new_obs_y - nn_y;

            //Calculate Gaussian
            gauss *= exp(-(pow(x_diff, 2)/p + pow(y_diff, 2)/q)) / r;
        }
        particles[i].weight = gauss;
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<int> distribution(weights.begin(), weights.end());
	vector<Particle> resample_particles;

	for(int i=0; i<num_particles; i++){
		resample_particles.push_back(particles[distribution(gen)]);	
	}

	particles = resample_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
