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
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

	default_random_engine gen;

	// Create normal distributions for x, y and theta
	normal_distribution<double> N_x(x, std[0]);
	normal_distribution<double> N_y(y, std[1]);
	normal_distribution<double> N_theta(theta, std[2]);

	// Generate particles
	particles.clear();
	for (int i = 0; i < num_particles; ++i) {
		Particle particle;

		particle.id = i;
		particle.x = N_x(gen);
		particle.y = N_y(gen);
		particle.theta = N_theta(gen);
		particle.weight = 0.5;
		particle.associations.clear();
		particle.sense_x.clear();
		particle.sense_y.clear();

		particles.push_back(particle);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
		double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	// Create normal distributions for the noises of x, y and theta
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	// Predict particles
	for (int i = 0; i < num_particles; ++i) {

		// The predicted x, y and theta
		double new_x, new_y, new_theta;
		// if theta is not zero
		if (abs(yaw_rate) >= 0.00001) {
			new_x = particles[i].x
					+ velocity / yaw_rate
							* (sin(particles[i].theta + yaw_rate * delta_t)
									- sin(particles[i].theta));
			new_y = particles[i].y
					+ velocity / yaw_rate
							* (cos(particles[i].theta)
									- cos(
											particles[i].theta
													+ yaw_rate * delta_t));
			new_theta = particles[i].theta + yaw_rate * delta_t;
		} else { // else theta is zero
			new_x = particles[i].x
					+ velocity * cos(particles[i].theta) * delta_t;
			new_y = particles[i].y
					+ velocity * sin(particles[i].theta) * delta_t;
			new_theta = particles[i].theta;
		}

		// add noise
		particles[i].x = new_x + N_x(gen);
		particles[i].y = new_y + N_y(gen);
		particles[i].theta = new_theta + N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
		std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (auto obs : observations) {

		double distance = 99999999;
		for (auto pre : predicted) {
			if (dist(obs.x, obs.y, pre.x, pre.y) < distance) {
				distance = dist(obs.x, obs.y, pre.x, pre.y);
				obs.id = pre.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

	weights.clear();
	for (auto& P : particles) {
		// Convert the coordinates of observations to global ones
		std::vector<LandmarkObs> actual_observations;

		actual_observations.clear();
		for (auto car_cord : observations) {
			LandmarkObs global_cord;
			global_cord.x = P.x + car_cord.x * cos(P.theta)
					- car_cord.y * sin(P.theta);
			global_cord.y = P.y + car_cord.x * sin(P.theta)
					+ car_cord.y * cos(P.theta);
			global_cord.id = car_cord.id;
			actual_observations.push_back(global_cord);
		}

		// Calculate predicted observations
		std::vector<LandmarkObs> predicted_observations;

		predicted_observations.clear();
		for (auto landmark : map_landmarks.landmark_list) {
			// Check if a landmark can be sensed
			if (dist(P.x, P.y, landmark.x_f, landmark.y_f) <= sensor_range) {
				LandmarkObs mark;
				// add landmark noise
				mark.id = landmark.id_i;
				mark.x = landmark.x_f;
				mark.y = landmark.y_f;
				predicted_observations.push_back(mark);
			}
		}

		// Initialize weight
		P.weight = 1.0;
		// update weight
		for (auto actual_observation : actual_observations) {
			// initialize weight
			// Calculate the nearest neighbour
			LandmarkObs neighbour;

			// compute the nearest predicted_observation to the actual_observation
			double distance = 99999999;

			for (auto pred_obs : predicted_observations) {
				if (dist(actual_observation.x, actual_observation.y, pred_obs.x,
						pred_obs.y) < distance) {
					distance = dist(actual_observation.x, actual_observation.y,
							pred_obs.x, pred_obs.y);
					neighbour.id = pred_obs.id;
					neighbour.x = pred_obs.x;
					neighbour.y = pred_obs.y;
				}
			}

			// update the weight
			P.weight *= 1 / (std_landmark[0] * sqrt(M_PI * 2))
					* exp(
							-((neighbour.x - actual_observation.x)
									* (neighbour.x - actual_observation.x))
									/ (2 * std_landmark[0] * std_landmark[0]))
					* 1 / (std_landmark[1] * sqrt(M_PI * 2))
					* exp(
							-((neighbour.y - actual_observation.y)
									* (neighbour.y - actual_observation.y))
									/ (2 * std_landmark[1] * std_landmark[1]));
		}
		weights.push_back(P.weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> d(weights.begin(), weights.end());
	std::vector<Particle> particles_new;
	for (int n = 0; n < num_particles; ++n) {
		particles_new.push_back(particles[d(gen)]);
	}
	particles = particles_new;
}

Particle ParticleFilter::SetAssociations(Particle particle,
		std::vector<int> associations, std::vector<double> sense_x,
		std::vector<double> sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	// remove trailing space
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	// remove trailing space
	s = s.substr(0, s.length() - 1);
	return s;
}
string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	// remove trailing space
	s = s.substr(0, s.length() - 1);
	return s;
}

