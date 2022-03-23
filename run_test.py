""""
Visual detection task. 
Based on 2AFC signal detection task by Stijn Nuiten.

Created by Anne Zonneveld, Feb 2022
"""

import os, sys, datetime

PYTHONPATH = '/opt/anaconda3/envs/psychopy/lib/python3.6'
sys.path.append('/opt/anaconda3/envs/psychopy/lib/python3.6')

import subprocess
import pickle, datetime, time
import numpy as np
from math import *
from IPython import embed as shell
import shutil
import glob
import pandas as pd 
import random
from psychopy import logging, visual, clock, event, core
from psychopy.tools.attributetools import attributeSetter, setAttribute
from psychopy.visual import GratingStim, TextStim, ImageStim, NoiseStim, DotStim, Window

logging.console.setLevel(logging.CRITICAL)

wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/'
sys.path.append(os.path.join(wd, 'exptools'))

import exptools
from exptools.core.trial import Trial
from exptools.core.session import Session

# import exptools2
# from exptools2.core.trial import Trial
# from exptools2.core.session import Session

p = ['FA', 'MISS']

fullscr = True
# total_trials = {'practice': 10,
# 				'actual': 30} # total should be dividable by 10 and 2
# 


trial_file = pd.read_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))   
# total_trials = len(trial_file)
total_trials  = 18000
runs = 4
nr_trials = int(total_trials / runs)
block_length = 500
nr_blocks = int(nr_trials/block_length)

target_categories = ["vegetable", "fruit", "drink", "insect", "bird", "clothing", "musical instrument", "body part", "plant", "sports equipment"]
mask_categories = ["furniture"]


class DetectTrial(Trial):
	def __init__(self, parameters = {}, phase_durations = [], session=None, screen=None, ID=0, categories = []):
		self.screen = screen
		self.parameters = parameters
		self.ID = ID
		self.phase_durations = phase_durations  
		self.session = session
		self.categories = categories
		self.block = np.floor(self.ID/self.session.block_length)
		self.create_stimuli()

		if self.ID == 0:
			self.session_start_time = clock.getTime()

		self.run_time = 0.0
		self.prestimulation_time = self.delay_1_time = self.image_stim_time = self.mask_stim_time = self.wait_time = self.answer_time =  0.0 
		self.parameters.update({'answer' :  -1,
								'correct': -1,
								'block': self.block,
								'RT' : 0
								})
		
		self.stopped = False
		super(
			DetectTrial,
			self).__init__(
			phase_durations=phase_durations,
			parameters = parameters,
			screen = self.screen,
			session = self.session			
			)
	
	def create_stimuli(self):
		self.center = ( self.screen.size[0]/2.0, self.screen.size[1]/2.0 )
		self.fixation = GratingStim(self.screen, tex='sin', mask = 'circle',size=6, pos=[0,0], sf=0, color ='black')

		# Determine messages
		if self.ID % self.session.block_length == 0 and self.ID > 0:
			perf = np.array(self.session.corrects)[-self.session.block_length:][np.array(self.session.corrects)[-self.session.block_length:] >= 0].sum() / float(self.session.block_length) * 100.0
			misses = (np.array(self.session.answers)[-self.session.block_length:]==-1).sum()
			intro_text = """\nBlock %i: %i%% correct!\n You missed %i out of %i trials. \nPress space to continue.""" % (self.block,perf, misses , block_length)
			print("perf: %i" %(perf))
		# elif self.ID == total_trials['practice'] + total_trials['actual']: #?
		# 	intro_text = """You have reached the end of the experiment. Thank you for participating."""
		else:
			intro_text = """During this experiment you will be presented with images of different categories. After image presentation, you will be asked whether the shown image belongs to a particular category. Press j (right) to answer yes or press f (left) to answer no. 
\n If the instructions are clear, press space to continue."""

		# Determine probe --> higher category or concept?
		if self.parameters['valid_cue'] == 1: 
			probed_category = self.parameters['category']
		else:
			# pick random invalid category (or should categories be evenly misrepresented?) + do not include mask categories
			# probed_category = random.choice([i for i in self.categories != self.parameters['category']])  
			probed_category = random.choice([i for i in target_categories != self.parameters['category']])  

		probe_text = """Did you see : %s? \n  Y (press j)/N (press f)""" % (probed_category)	
		# practice_text = """You will first start with a practice block. During this block you will get immediate feedback. If your given answer
		# was correct, you will see a blue cue. If incorrect, you will see a red cue. \n Press space to start the practice block"""

		self.message = TextStim(self.screen, pos=[0,0],text= intro_text, color = (1.0, 1.0, 1.0), height=20)
		self.image_stim = ImageStim(self.screen, pos=[0,0], image = self.parameters['target_path'])
		self.probe = TextStim(self.screen, pos=[0,0], text = probe_text, color = (1.0, 1.0, 1.0), height=20)
		self.mask_stim = ImageStim(self.screen, pos = [0,0], image = self.parameters['mask_path'])

	
	def draw(self):
		# draw additional stimuli:

		if self.phase == 0: # instruction / message
			# if self.ID == 0 or self.ID % self.session.block_length == 0 or self.ID == total_trials['practice'] + total_trials['actual']:
			if self.ID == 0 or self.ID % self.session.block_length == 0:
				self.message.draw()
			else:
				self.fixation.draw()

		if self.phase == 1: # pre-image fixation
			self.fixation.color = 'black'
			self.fixation.draw()
		
		if self.phase == 2:  # image presentation
			self.image_stim.draw()

		if self.phase == 3: # mask presentation
			self.mask_stim.draw()
			
		if self.phase == 4: # empty screen
			pass
		
		if self.phase == 5: # prompt
			self.probe.draw()

		# elif self.phase == 6: # feedback / only in practice block
		# 	if self.ID <= total_trials['practice']:
		# 		if self.parameters['correct'] == 1 :
		# 			self.fixation.color = 'blue'
		# 		elif self.parameters['correct'] == 0:
		# 			self.fixation.color = 'red'
		# 	else:
		# 		self.fixation.color = 'black'
			
		# 	self.fixation.draw()

		super(DetectTrial, self).draw()

	def event(self):
		trigger = None

		for ev in event.getKeys():
			if len(ev) > 0:
				if ev in ['esc', 'escape']:
					self.events.append(
						[-99, clock.getTime() - self.start_time])
					self.stopped = True
					self.session.stopped = True
					print('run canceled by user')

				elif ev == 'space':
					self.events.append(
						[99, clock.getTime() - self.start_time])
					if self.phase == 0:
						self.phase_forward()

				elif ev == 'j':
					self.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 5:
						self.parameters.update({'answer':1})
						if self.parameters['valid_cue'] == self.parameters['answer']:
							self.parameters['correct'] = 1
						else:
							self.parameters['correct'] = 0
						self.phase_forward()

				elif ev == 'f':
					self.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 5:
						self.parameters.update({'answer':0})
						if self.parameters['valid_cue'] == self.parameters['answer']:
							self.parameters['correct'] = 0 
						else:
							self.parameters['correct'] = 1
						self.phase_forward()

			super(DetectTrial, self).key_event( event )

	def run(self):
		super(DetectTrial, self).run()

		while not self.stopped:
			self.run_time = clock.getTime() - self.start_time
							
			if self.phase == 0:
				self.prestimulation_time = clock.getTime()

				# For all trials that are not FTIB, skip phase 0
				if self.ID != 0 and self.ID % self.session.block_length != 0:
					if (self.prestimulation_time  - self.start_time ) > self.phase_durations[0]:
						self.phase_forward()
				
			elif self.phase == 1:  # pre-stim cue; phase is timed
				self.delay_1_time = clock.getTime()
				if ( self.delay_1_time - self.prestimulation_time ) > self.phase_durations[1]:
					self.phase_forward()

			elif self.phase == 2:  # image presentation; phase is timed
				self.image_stim_time = clock.getTime()				
				if ( self.image_stim_time - self.delay_1_time ) > self.phase_durations[2]: 
					self.phase_forward()

			elif self.phase == 3: # mask presentation; phase is timed
				self.mask_stim_time = clock.getTime()
				if (self.mask_stim_time - self.image_stim_time) > self.phase_durations[3]:
					self.phase_forward

			elif self.phase == 4: # wait; blank screen; phase timed
				self.wait_time = clock.getTime()
				if (self.wait_time - self.mask_stim_time) > self.phase_durations[4]:
					self.phase_forward

			elif self.phase == 5:   # Decision interval; phase is timed, but aborted at response
				self.answer_time = clock.getTime()
				if self.parameters['answer'] != -1: #end phase when respond
					# get reaction time
					self.parameters['RT'] = self.answer_time - self.wait_time
					self.stopped = True
					self.stop()
					return
				if ( self.answer_time  - self.wait_time) > self.phase_durations[5]: #end phase after some time when no response
					self.parameters['RT'] = float("nan")
					self.stopped = True
					self.stop()
					return

			# elif self.phase == 6:				# Feedback; only pratice block
			# 	self.delay_4_time = clock.getTime()	
			# 		if (self.delay_4_time - self.answer_time) > self.phase_durations[5]:
			# 			self.stopped = True
			# 			self.stop()
			# 			return
			
			# events and draw:
			self.event()
			self.draw()
						
		# we have stopped:
		# frame_timer.log_histogram()     
		self.stop()


class DetectSession(Session):
	def __init__(self, subject_nr, nr_trials, block_length, index_number=1):
		super(DetectSession, self).__init__(subject_nr, index_number)
		self.create_screen(size=[1920, 1080],full_screen = fullscr, background_color = (0.5, 0.5, 0.5), physical_screen_distance = 80, engine = 'pygaze') 
		self.block_length = block_length
		self.nr_trials = nr_trials
		self.index_number = index_number
		self.standard_parameters = {'run': self.index_number}

		self.create_output_filename() # standard function?
		# self.load_THINGS()
		self.determine_trials()
		self.create_yes_no_trials()
	
	# def create_output_filename(self):
	# 	data_path = os.path.join(wd, "data")
	# 	if not os.path.exists(os.path.join(data_path, "sub_" + str(subject_nr))):
	# 		os.makedirs(os.path.join(data_path, "sub_" + str(subject_nr)))
	# 	self.output_path = os.path.join(data_path, "sub_" + str(subject_nr))
	# 	self.output_file = os.path.join(data_path, "sub_" + str(subject_nr), "run_" + str(self.index_number))

	# def load_THINGS(self):
	# 	self.THINGS_data = pd.read_csv(os.path.join(wd, "help_files", "selection_THINGS.csv" ), sep=',', header=0)
	# 	self.categories = pd.unique(self.THINGS_data['category'])
	# 	self.concepts = pd.unique(self.THINGS_data['concept'])

	def determine_trials(self):
		# self.images = []
		# self.nr_per_cat = int(total_trials['actual'] / len(self.categories)) # or self.nr_trials
		# for category in self.categories:
		# 	selection = random.sample(self.THINGS_data[self.THINGS_data['category'] == category].values.tolist(), self.nr_per_cat)
		# 	self.images.append(selection)
		# self.images = np.array(self.images)


		self.output_path = os.path.join(wd, "data", "sub_" + str(subject_nr))
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path)  

		# deterimine indexes of previous runs
		filter = np.ones(total_trials, dtype=bool)
		if self.index_number > 1:
			previous_index = pd.read_csv(os.path.join(self.output_path, "previous_trials.csv"))
			for i in previous_index['index']:
				filter[i] = False

		# pick trials for current run
		self.all_trials_index = np.arange(total_trials)
		self.available_trials_index = self.all_trials_index[filter]
		self.current_trials_index = random.sample(self.available_trials_index.tolist(), nr_trials)
		
		# save chosen indices to be run in previous_trials.csv --> maybe later? 
		df = pd.DataFrame(self.current_trials_index, columns=['index'])
		if self.index_number > 1:
			df = df.append(previous_index, ignore_index=True)

		# export 
		# fix export
		df.to_csv(os.path.join(self.output_path, 'previous_trials.csv')) 


	def create_yes_no_trials(self):
		"""creates trials for yes/no runs"""
					
		# Create yes-no trials in nested for-loop:
		self.valid_cue = np.array([0,1])
		self.trial_parameters_and_durs = []    
		trial_counter = 0
		self.total_duration = 0

		phase_durs = [-0.01, 0.5, 0.024, 0.072, 0.75, 1.2]

		# Loop over all trials
		for i in range(int(self.nr_trials/self.valid_cue.shape[0])):
			for j in range(self.valid_cue.shape[0]):

				# Pick trial
				trial_index = self.current_trials_index[trial_counter]
				trial_info = trial_file.iloc[trial_index]

				# Update params with all trial info
				params = self.standard_parameters
				params.update({ 'valid_cue': self.valid_cue[j],
								'index': trial_info[0],
								'target_path': trial_info['ImageID'],
								'concept': trial_info['concept'],
								'category': trial_info['category'],
								'mask_type' : trial_info['mask_type'],
								'mask_path': trial_info['mask']
				}) 

				self.trial_parameters_and_durs.append([params.copy(), np.array(phase_durs)])
				self.total_duration += np.array(phase_durs).sum()
				trial_counter += 1

		# self.run_order = np.argsort(np.random.rand(len(self.trial_parameters_and_durs))) # double random
		self.run_order = np.array(self.current_images_index)

		# print params:
		print("number trials: %i." % trial_counter)
		if trial_counter != nr_trials:
			raise ValueError('number of created trials does not match pre-defined number of trials')

		print("total duration: %.2f min." % (self.total_duration / 60.0))

	def run(self):
		"""run the session"""
		# cycle through trials
		self.corrects = []
		self.answers = []
		self.clock = clock

		self.start_time= clock.getTime()
		trial_counter = 0

		while trial_counter < self.nr_trials:
			print("trial count: %i" %(trial_counter))
			this_trial = DetectTrial(parameters=self.trial_parameters_and_durs[self.run_order[trial_counter]][0], phase_durations=self.trial_parameters_and_durs[self.run_order[trial_counter]][1], session=self, screen=self.screen, ID=trial_counter, categories = self.categories)
			this_trial.run()	

			self.corrects.append(this_trial.parameters['correct'])
			self.answers.append(this_trial.parameters['answer'])

			if self.stopped == True:
				break
			trial_counter += 1
		
		
		lastPerf = np.array(self.corrects)[-self.block_length:][np.array(self.corrects)[-self.block_length:] >= 0].sum() / float(self.block_length) * 100.0
		lastMisses = (np.array(self.answers)[-self.block_length:]==-1).sum()	
		self.stop_time = clock.getTime()

		#print 'elapsed time: %.2fs' %(self.stop_time-self.start_time)
		#print 'performance: %.2f' %(lastPerf)
		#print 'misses: %.2f' %(lastMisses)
		
		self.screen.clearBuffer

		self.close()

def main(subject_nr, index_number, nr_trials):
    
    ts = DetectSession(subject_nr = subject_nr, nr_trials=nr_trials, block_length = block_length, index_number = index_number)
    ts.run()

if __name__ == '__main__':
	# Store info about the experiment session
	subject_nr = str(input('Participant nr: '))

	index_number = int(input('Which run: ')) 

	main(subject_nr = subject_nr, index_number=index_number, nr_trials=nr_trials)