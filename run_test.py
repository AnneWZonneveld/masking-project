""""
Visual detection task. 
Based on 2AFC signal detection task by Stijn Nuiten.
Created by Anne Zonneveld, Feb 2022
"""

import os, sys, datetime
import subprocess
import pickle, datetime, time
import numpy as np
from math import *
from IPython import embed as shell
import shutil
import glob
import pandas as pd 
import random
import scipy
from psychopy import logging, visual, clock, event, core
from psychopy.tools.attributetools import attributeSetter, setAttribute
from psychopy.visual import GratingStim, TextStim, ImageStim, NoiseStim, DotStim, Window, TextBox2
from PIL import Image

# sys.path.append(os.path.join('C:\\', 'Users', 'onderzoekl210','Desktop', 'Anne', 'masking-project', 'exptools'))
wd = '/Users/AnneZonneveld/Documents/STAGE/masking-project/'
# wd = os.path.join('C:\\','Users', 'onderzoekl210','Desktop', 'Anne', 'masking-project')

log_dir = os.path.join(wd, 'logfiles')
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

logging.console.setLevel(logging.WARNING)
# logging.console.setLevel(logging.EXP)

sys.path.append(os.path.join(wd, 'exptools'))

import exptools
from exptools.core.trial import Trial
from exptools.core.session import Session


fullscr = False
screen_size  = [1920, 1080]

trial_file = pd.read_csv(os.path.join(wd, 'help_files', 'selection_THINGS.csv'))   
concept_file = pd.read_csv(os.path.join(wd, 'help_files', 'concept_selection.csv'), header=0, sep=';')

runs = 3
nr_trials = 1944
block_length = 54 #divideable of total_trials
nr_blocks = int(nr_trials/block_length) # 37

#runs = 2
#nr_trials = 20
#block_length = 5 #divideable of total_trials
#nr_blocks = int(nr_trials/block_length)

target_categories = pd.unique(trial_file['category']).tolist()
target_concepts = pd.unique(trial_file['concept']).tolist()

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
								'RT' : 0,
								'target_onset': 0,
								'target_offset': 0, 
								'mask_onset': 0,
								'mask_offset': 0, 
								'probe_onset': 0,
								'response_time': 0
								})
		

		self.target_drawn = self.mask_drawn = self.probe_drawn = 0
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
			intro_text = """\nBlock %i out %i \n %i%% correct!\n Press space to continue.""" % (self.block, nr_blocks, perf)
			print("perf: %i" %(perf))
		else:
			intro_text = """During this experiment you will be presented with series of images. The first image in a trial is the target and the second image a mask.
			\n After presentation, you have to answer whether the target image belongs to the probed category. Press l (right) to answer yes or press a (left) to answer no. 
\n If the instructions are clear, press space to start."""
			
		# Determine probe 
		if self.parameters['valid_cue'] == 1: 
			probed_concept = self.parameters['concept']
		else:
			other_concepts = []
			for i in range(len(concept_file)):
				if concept_file['category'].iloc[i] != self.parameters['category']:
					other_concepts.append(concept_file['concept'].iloc[i])
			probed_concept = random.choice(other_concepts)  
		
		self.parameters.update({'probe':probed_concept})
		probed_concept = probed_concept.replace('_', ' ')

		# probe_text = """Did you see : '%s'? \n  NO (press f) / YES (press j)""" % (probed_concept)
		probe_text = """%s""" % (probed_concept)
		instruction_text = """NO (press a) / YES (press l)"""
		outro_text = """This is the end of today's session. Thank you for participating!"""	

		self.message = TextStim(self.screen, pos=[0,0],text= intro_text, color = (1.0, 1.0, 1.0), height=20)
		self.image_stim = ImageStim(self.screen, pos=[0,0], image = self.parameters['target_path'])
		# self.probe = TextStim(self.screen, pos=[0,0], text = probe_text, color = (1.0, 1.0, 1.0), height=20)
		self.probe = TextBox2(self.screen, pos=[0,0], text = probe_text, color = (1.0, 1.0, 1.0), font='Arial', bold = True, letterHeight=50, alignment='center', size=[None, None])
		self.instruction= TextBox2(self.screen, pos=[0, -70], text = instruction_text, color = (1.0, 1.0, 1.0), font='Arial', letterHeight = 20, alignment='center', size=[None, None])
		self.outro = TextStim(self.screen, pos=[0,0], text = outro_text, color = (1.0, 1.0, 1.0), height=20)

		if self.parameters['mask_path'] != 'no_mask':
			self.mask_stim = ImageStim(self.screen, pos = [0,0], image = self.parameters['mask_path'])
		else:
			self.mask_stim = self.fixation

		super(DetectTrial, self).draw()

	def event(self):
		trigger = None

		for ev in event.getKeys():
			if len(ev) > 0:
				if ev in ['esc', 'escape']:
					self.session.events.append(
						[-99, clock.getTime() - self.start_time])
					self.stopped = True
					self.session.stopped = True
					print('run canceled by user')
					self.session.close()

				elif ev == 'space':
					self.session.events.append(
						[99, clock.getTime() - self.start_time])
					if self.phase == 0:
						self.phase_forward()
					if self.phase == 6:
						self.stopped = True
						self.session.stopped = True
						print("End of session")

				elif ev == 'l':
					self.session.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 5:
						# yes 
						self.parameters.update({'answer':1, 'response_time': clock.getTime()-self.session.start_time})
						if self.parameters['valid_cue'] == self.parameters['answer']:
							self.parameters['correct'] = 1
						else:
							self.parameters['correct'] = 0
						self.parameters['RT'] = self.parameters['response_time'] - self.parameters['probe_onset']
						if self.ID != (self.session.nr_trials - 1):
							self.stopped = True
							self.stop()
						else:
							self.phase_forward()


				elif ev == 'a':
					self.session.events.append([1,clock.getTime()-self.start_time])
					if self.phase == 5:
						# no
						self.parameters.update({'answer':0, 'response_time': clock.getTime()-self.session.start_time})
						if self.parameters['valid_cue'] == self.parameters['answer']:
							self.parameters['correct'] = 1
						else:
							self.parameters['correct'] = 0
						self.parameters['RT'] = self.parameters['response_time'] - self.parameters['probe_onset']
						if self.ID != (self.session.nr_trials - 1):
							self.stopped = True
							self.stop()
						else:
							self.phase_forward()

			super(DetectTrial, self).key_event( event )

	def run(self):
		super(DetectTrial, self).run()

		phase_5_counter = 0
		
		while not self.stopped:

			self.run_time = clock.getTime() - self.start_time

			if self.phase == 0:
				print('phase 0')
				
				if self.ID == 0 or self.ID % self.session.block_length == 0:
					print('FTIB')
					self.message.draw()
					self.screen.flip()
				else:
					self.phase_forward()
					
			elif self.phase == 1:  # pre-stim cue; phase is timed
				print('phase 1')
				self.fixation.color = 'black'
				for frameN in range(int(self.phase_durations[1])):
					self.fixation.draw()
					self.screen.flip()

				if frameN == int(self.phase_durations[1]) - 1:
					self.phase_forward()

			elif self.phase == 2:  # image presentation; phase is timed
				print('phase 2')
				for frameN in range(int(self.phase_durations[2])):
					self.image_stim.draw()
					if frameN == 0:
						self.parameters.update({'target_onset': clock.getTime() - self.session.start_time})
					self.screen.flip()

				if frameN == int(self.phase_durations[2]) - 1:
					self.parameters.update({'target_offset': clock.getTime() - self.session.start_time})
					self.phase_forward()
								
			elif self.phase == 3: # mask presentation; phase is timed
				print('phase 3')
				for frameN in range(int(self.phase_durations[3])):
					self.mask_stim.draw()
					if frameN == 0:
						self.parameters.update({'mask_onset': clock.getTime() - self.session.start_time})
					self.screen.flip()

				if frameN == int(self.phase_durations[3]) - 1:
					self.parameters.update({'mask_offset': clock.getTime() - self.session.start_time})
					self.phase_forward()

			elif self.phase == 4: # wait; blank screen; phase timed
				print('phase 4')
				for frameN in range(int(self.phase_durations[4])):
					self.fixation.draw()
					self.screen.flip()

				if frameN == int(self.phase_durations[4])-1:
					# self.wait_time = clock.getTime()
					self.phase_forward()

			elif self.phase == 5:   #Probe, Aborted at key response (a or l)
				print('phase 5')
				self.probe.draw()
				self.instruction.draw()
				if phase_5_counter == 0:
					print('counter = 0')
					self.parameters.update({'probe_onset': clock.getTime() - self.session.start_time})
				phase_5_counter += 1
				self.screen.flip()

			# events and draw:
			self.event()
					

class DetectSession(Session):
	def __init__(self, subject_nr, nr_trials, block_length, index_number=1):
		super(DetectSession, self).__init__(subject_nr, index_number)
		self.create_screen(size=screen_size,full_screen = fullscr, background_color = (0.5, 0.5, 0.5), physical_screen_distance = 80, engine = 'pygaze') 
		self.screen.mouseVisible = False
		self.block_length = block_length
		self.nr_trials = nr_trials
		self.index_number = index_number
		self.standard_parameters = {'run': self.index_number}

		self.create_output_filename() # standard function?
		self.create_yes_no_trials()
	
	
	def create_yes_no_trials(self):
		"""creates trials for yes/no runs"""

		self.all_trials_index = np.arange(nr_trials)
		self.current_trials_index = random.sample(self.all_trials_index.tolist(), nr_trials)

		# Create yes-no trials in nested for-loop:
		self.valid_cue = np.array([0,1])
		self.trial_parameters_and_durs = []    
		trial_counter = 0
		self.total_duration = 0

		# phase_durs = [-0.01, 0.5, 0.024, 0.072, 0.75, 1.2]
		# phases in frames (except for phase 0 and 5)
		phase_durs = [-0.01, 60, 2, 6, 90, 1.2]

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
								'mask_path': trial_info['mask_path']
				}) 

				self.trial_parameters_and_durs.append([params.copy(), np.array(phase_durs)])
				self.total_duration += np.array(phase_durs).sum()
				trial_counter += 1

		# self.run_order = np.argsort(np.random.rand(len(self.trial_parameters_and_durs))) # double random
		# self.run_order = np.array(self.current_trials_index)

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
			# this_trial = DetectTrial(parameters=self.trial_parameters_and_durs[self.run_order[trial_counter]][0], phase_durations=self.trial_parameters_and_durs[self.run_order[trial_counter]][1], session=self, screen=self.screen, ID=trial_counter, categories = target_categories)
			this_trial = DetectTrial(parameters=self.trial_parameters_and_durs[trial_counter][0], phase_durations=self.trial_parameters_and_durs[trial_counter][1], session=self, screen=self.screen, ID=trial_counter, categories = target_categories)
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
	subject_nr = 0
    #subject_nr = str(input("Participant nr: "))

	index_number = 1
    #index_number = int(input("Which run: ")) 

	log_filename = os.path.join(log_dir, "s{}_r{}".format(subject_nr, index_number))
	lastLog = logging.LogFile(log_filename, level=logging.INFO, filemode='w')

	main(subject_nr = subject_nr, index_number=index_number, nr_trials=nr_trials)