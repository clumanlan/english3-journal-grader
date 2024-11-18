import streamlit as st
import json
import re
from typing import List, Dict
import boto3 
from io import StringIO, BytesIO
from PIL import Image
import pandas as pd
from datetime import datetime

from haystack.components.generators import OpenAIGenerator

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
AWS_ACCESS_KEY_ID = st.secrets['AWS_DYNAMODB_ACCESS_KEY']
AWS_SECRET_ACCESS_KEY = st.secrets['AWS_DYNAMODB_SECRET']


gpt4_client = OpenAIGenerator(model="gpt-4o")

@st.cache_data
def get_student_names():

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-2')
    
    s3 = session.client('s3')

    response = s3.get_object(Bucket='aplit-journal-grader', Key='english3_student.csv')
    csv_content = response['Body'].read().decode('utf-8')
    student_df = pd.read_csv(StringIO(csv_content))
    
    return student_df['Name']

@st.cache_data
def get_msfrate_picture():

    s3_resource = boto3.resource(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='us-east-2', 
        service_name='s3')

    bucket = s3_resource.Bucket('aplit-journal-grader')
    image = bucket.Object('ms_frate_photo.PNG')

    img_data = image.get().get('Body').read()
    image = Image.open(BytesIO(img_data))

    return image


def main_claim_critique(prompt, main_claim):
        
        main_claim_score_rubric = """
            Score of 4: 
            - Clear and easy to understand.
            - Concise.
            - Includes accurate authorial choice.
            - Includes accurate author’s purpose.


            Score of 3: Too long, too much information
            OR
            Vague or not specific enough to fully understand argument 

            Score of 2: Missing or incorrect authorial choice
            OR
            Missing or incorrect author’s purpose.

            Score of 1: Missing or incorrect authorial choice or 
            Missing or incorrect author’s purpose

            """
        
        main_claim_prompt = f"""
            You are an English highschool teacher for juniors that are currently reading at a 7th grade level, you provide student this prompt: {prompt}.
            The student, in return, provides this thesis claim in response: {main_claim}.
            Provide a score and feedback in bulleted notes based on this rubric: {main_claim_score_rubric}.
            Provide helpful and encouraging feedback that is tailored to a student at 7th grade reading level.
            Provide it in the format: 
                Score: 
                Suggestions:
            """
        
        return gpt4_client.run(main_claim_prompt)['replies'][0]


def evidence_one_critique(main_claim, evidence):
    
    evidence_score_rubric = """
        Score of 4: 
        You have a direct quote 
        Quote fully supports your claim, and 
        Only the necessary parts of the quote are used
        Quote has the context needed for the reader to understand the quote.

        Score of 3:
        No context, or unclear context
        OR
        Too long, difficult to identify which parts are significant 
        OR 
        Quote not embedded into the sentence

        Score of 2: 
        Paraphrase instead of quote
        OR 
        Tangential to the claim (related to the claim, but does not directly prove it)

        Score of 1: 
        Evidence is entirely unrelated to the claim,
        OR
        Evidence is paraphrased poorly, making the evidence too confusing to contribute to the argument

        """


    evidence_prompt = f"""
        You are an English highschool teacher for juniors that are currently reading at a 7th grade level,  
        the student has made this claim: {main_claim}.
        To support this claim the student provides two pieces of evidence one piece of the evidence is: {evidence}.
        Please provide a grade and helpful suggestions to get higher score in bulleted notes based on this rubric: {evidence_score_rubric}.
        Provide helpful and encouraging feedback that is tailored to a student at 7th grade reading level.
        Please provide it in the format: 
            Score: 
            Suggestions:
        """

    return gpt4_client.run(evidence_prompt)['replies'][0]


def evidence_two_critique(main_claim, evidence_one, evidence_two):
    
  evidence_score_rubric = """
        Score of 4: 
        You have a direct quote 
        Quote fully supports your claim, and 
        Only the necessary parts of the quote are used
        Quote has the context needed for the reader to understand the quote.

        Score of 3:
        No context, or unclear context
        OR
        Too long, difficult to identify which parts are significant 
        OR 
        Quote not embedded into the sentence

        Score of 2: 
        Paraphrase instead of quote
        OR 
        Tangential to the claim (related to the claim, but does not directly prove it)

        Score of 1: 
        Evidence is entirely unrelated to the claim,
        OR
        Evidence is paraphrased poorly, making the evidence too confusing to contribute to the argument

        """
  evidence_prompt = f"""
        You are an English highschool teacher for juniors that are currently reading at a 7th grade level,  the student has made this claim: {main_claim}.
        To support this claim the student provides two pieces of evidence. 
        The first piece of evidence is: {evidence_one}.
        The second piece of evidence: {evidence_two}.
        Please be sure to directly label what number piece evidence you're referring to.
        Please provide a grade and helpful suggestions to get higher score in bulleted notes based on this rubric: {evidence_score_rubric}.
        You will only provide feedback on the second piece of evidence.
        Provide helpful and encouraging feedback that is tailored to a student at 7th grade reading level.
        Please provide it in the format: 
            Score: 
            Critique: 
            Suggestions:
        """
  
  return gpt4_client.run(evidence_prompt)['replies'][0]


def reasoning_critique(main_claim, evidence, reasoning):
    
    reasoning_score_rubric = """
        Score of 4: Identifies the significant aspects of the evidence.
        Clearly identifies the authorial choice.
        Explains how the evidence proves the author’s purpose.
        
        Score of 3: Two of the criteria of score of 4 are met.

        Score of 2: One of the criteria of score of 4 are met.

        Score of 1: None of the criteria of score of 4 are met.

        """

    reasoning_prompt = f"""
        You are an English highschool teacher for juniors that are currently reading at a 7th grade level, , the student has provided this evidence: {evidence}.
        The student's main thesis: {main_claim}. To prove their thesis the student has provided this reasoning: {reasoning}.
        Please provide a grade and helpful suggestions to get higher score in bulleted notes based on this rubric: {reasoning_score_rubric}.
        Provide helpful and encouraging feedback that is tailored to a student at 7th grade reading level.
        Please provide it in the format: 
            Score: 
            Suggestions:
        
        """
    
    return gpt4_client.run(reasoning_prompt)['replies'][0]
    

def synthesis_critique(prompt, main_claim, evidence_one, reasoning_one, evidence_two, reasoning_two, conculsion_statement):
    
    synthesis_score_rubric = """
        Score of 4: Connections are made between all points in the paragraph, 
        arrives at a final synthesis about the text based on synthesis, 
        and shows in-depth insight about the text.

        Score of 3: Connections are present but surface-level, 
        OR synthesis is present but surface-level.

        Score of 2: Connections are too vague, incorrect, or missing. 

        Score of 1: All parts are inaccurate, OR arguments are merely restated or summarized.
        """
    
    synthesis_prompt = f"""
        As if You are an English highschool teacher for juniors that are currently reading at a 7th grade level, you provide student this prompt: {prompt}.
        The student, in return, provides this thesis claim in response: {main_claim} and this evidence:
        {evidence_one} and reasoning: {reasoning_one}.
        Along with this evidence: {evidence_two} and reasoning: {reasoning_two}
        Please provide a grade and helpful suggestions to get higher score in bulleted notes of the synthesis: 
        {conculsion_statement} 
        Based on this rubric: 
        {synthesis_score_rubric}
        Only provide feedback on the syntehesis.
        Provide helpful and encouraging feedback that is tailored to a student at 7th grade reading level.
        Please provide it in the format: 
            Score: 
            Suggestions:

        """

    return gpt4_client.run(synthesis_prompt)['replies'][0]


# INITIALIZE SESSION STATE 
if 'student_name' not in st.session_state:
    st.session_state['student_name'] = ''

if 'current_section' not in st.session_state:
    st.session_state.current_section = 1

sections = [ "main_claim", "evidence_one", "reasoning_one", 
           "evidence_two", "reasoning_two", "synthesis"]

for section in sections:
    if f'{section}_content' not in st.session_state:
        st.session_state[f'{section}_content'] = ""
    if f'{section}_submitted' not in st.session_state:
        st.session_state[f'{section}_submitted'] = False
    if f'{section}_feedback' not in st.session_state:
        st.session_state[f'{section}_feedback'] = ""


col1, col2 = st.columns([1, 3])  
with col1:
    msfrate_pic = get_msfrate_picture()
    st.image(msfrate_pic)

with col2:
    st.subheader("Frate Train Collaborative Journal Response Grader")

# student_names = get_student_names() , *student_names
student_name = st.selectbox('Please start typing to select name...', ['', 'test'],   key='student_name')

st.divider()

prompt = """The Handmaid's Tale focuses on systemic political and social issues. 
            Write a paragraph about how Atwood uses authorial choices to explore a specific issue 
            and explain how the issue contributes to the meaning of the work as a whole."""

st.write('Reminder this is your prompt:')
st.write(prompt)

st.divider()

for section in sections:
    if st.session_state[f'{section}_submitted']:
        st.write(f"### {section.replace('_', ' ').title()}")
        st.text_area(
            "Content:",
            value=st.session_state[f'{section}_content'],
            key=f'display_{section}',
            disabled=True
        )
        if st.session_state[f'{section}_feedback']:
            st.write("Feedback:", st.session_state[f'{section}_feedback'])
        st.divider()


# Find the current (first unsubmitted) section
current_section = None
for section in sections:
    if not st.session_state[f'{section}_submitted']:
        current_section = section
        break

# Show current section if there is one
if current_section:
    st.write(f"### {current_section.replace('_', ' ').title()}")
    current_input = st.text_area(
        "Your response:",
        key=f'input_{current_section}'
    )
    
    if st.button(f"Submit {current_section.replace('_', ' ').title()}") :
        # Store the response
        st.session_state[f'{current_section}_content'] = current_input

        if current_section == 'main_claim':
            feedback = main_claim_critique(prompt, current_input)
        elif current_section == 'evidence_one':
            feedback = evidence_one_critique(st.session_state['main_claim_content'], 
                                             current_input)
        elif current_section == 'reasoning_one':
            feedback = reasoning_critique(st.session_state['main_claim_content'],  
                                          st.session_state['evidence_one_content'], 
                                          current_input)
        elif current_section == 'evidence_two':
            feedback = evidence_two_critique(st.session_state['main_claim_content'],  
                                          st.session_state['evidence_one_content'], 
                                          current_input)
        elif current_section == 'reasoning_two':
            feedback = reasoning_critique(
                                          st.session_state['main_claim_content'],  
                                          st.session_state['evidence_two_content'],  
                                          current_input)
        else: 
            feedback = synthesis_critique(prompt, 
                                          st.session_state['main_claim_content'],  
                                          st.session_state['evidence_one_content'], 
                                          st.session_state['reasoning_one_content'], 
                                          st.session_state['evidence_two_content'], 
                                          st.session_state['reasoning_two_content'], 
                                          current_input)


        st.session_state[f'{current_section}_feedback'] = feedback
        
        st.session_state[f'{current_section}_submitted'] = True
        
        st.rerun()
