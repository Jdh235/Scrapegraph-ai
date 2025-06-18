"""
MergeAnswersNode Module with relevance-ranked sources
"""

from typing import List, Optional, Dict, Any

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from ..prompts import TEMPLATE_COMBINED
from ..utils.output_parser import (
    get_pydantic_output_parser,
    get_structured_output_parser,
)
from .base_node import BaseNode


class MergeAnswersNode(BaseNode):
    """
    A node responsible for merging the answers from multiple graph instances into a single answer.
    Also ranks sources by relevance to the generated output.

    Attributes:
        llm_model: An instance of a language model client, configured for generating answers.
        verbose (bool): A flag indicating whether to show print statements during execution.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique identifier name for the node, defaulting to "MergeAnswers".
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "MergeAnswers",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)

        self.llm_model = node_config["llm_model"]

        if isinstance(self.llm_model, ChatOllama):
            if self.node_config.get("schema", None) is None:
                self.llm_model.format = "json"
            else:
                self.llm_model.format = self.node_config["schema"].model_json_schema()

        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        """
        Executes the node's logic to merge the answers from multiple graph instances into a
        single answer and ranks sources by relevance.

        Args:
            state (dict): The current state of the graph. The input keys will be used
                            to fetch the correct data from the state.

        Returns:
            dict: The updated state with the output key containing the generated answer
                  and ranked sources.

        Raises:
            KeyError: If the input keys are not found in the state, indicating
                      that the necessary information for generating an answer is missing.
        """

        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)

        input_data = [state[key] for key in input_keys]
        
        if self.verbose:
            print('merge answer used')
            print(input_data)

        user_prompt = input_data[0]
        answers = input_data[1]
        
        if self.verbose:
            print('answers')
            print(answers)
        
        # Get the URLs from the state
        urls = []
        if "urls" in state:
            urls = state["urls"]
        elif "considered_urls" in state:
            urls = state["considered_urls"]
            
        # Create a mapping of content to URLs if we have URLs
        content_url_map = {}
        if urls and len(urls) == len(answers):
            for i, answer in enumerate(answers):
                content_url_map[f"CONTENT WEBSITE {i + 1}"] = {
                    "content": answer,
                    "url": urls[i]
                }
        
        # Format the content for the prompt
        answers_str = ""
        for i, answer in enumerate(answers):
            answers_str += f"CONTENT WEBSITE {i + 1}: {answer}\n"

        # Modify the prompt to request relevance ranking
        if self.node_config.get("schema", None) is not None:
            if isinstance(self.llm_model, (ChatOpenAI, ChatMistralAI)):
                self.llm_model = self.llm_model.with_structured_output(
                    schema=self.node_config["schema"]
                )  # json schema works only on specific models

                output_parser = get_structured_output_parser(self.node_config["schema"])
                format_instructions = "NA"
            else:
                output_parser = get_pydantic_output_parser(self.node_config["schema"])
                format_instructions = output_parser.get_format_instructions()

        else:
            output_parser = JsonOutputParser()
            format_instructions = output_parser.get_format_instructions()
        
        # Add a request for relevance ranking to the template
        relevance_instruction = """
        Additionally, rank the sources by their relevance to your answer. 
        Include a 'source_relevance' field in your json response that lists the sources 
        in descending order of relevance. For each source, include its identifier 
        (e.g., "CONTENT WEBSITE 1", "CONTENT WEBSITE 2", etc.) and a relevance score from 1-10.
        Relevance score should be based on how much information is present in the source to your answer.
        For high relevance, the source should offer complete or near-complete coverage of your answer, and the article should be about the same topic as the user prompt.
        """
        
        # Append the relevance instruction to the format instructions
        extended_format_instructions = format_instructions + "\n" + relevance_instruction


        #alter this to add a new variable - full description of allegation. How do I pass new variable in?
        prompt_template = PromptTemplate(
            template=TEMPLATE_COMBINED,
            input_variables=["user_prompt"],
            partial_variables={
                "format_instructions": extended_format_instructions,
                "website_content": answers_str,
            },
        )

        merge_chain = prompt_template | self.llm_model | output_parser
        result = merge_chain.invoke({"user_prompt": user_prompt})
        
        # Process the source relevance rankings
        if 'source_relevance' in result:
            # Get the ordered list of source identifiers from the LLM response
            ranked_sources = result['source_relevance']

            print('ranked sources')
            print(ranked_sources)
            
            """ordered_urls = []
            for source_item in ranked_sources:
                print('source item')
                print(source_item)
                source_id = source_item.get('source')
                if source_id in content_url_map:
                    ordered_urls.append(content_url_map[source_id]['url'])
            print(ordered_urls)"""

            ordered_urls = []
            low_confidence_flag = False
            for i, source_item in enumerate(ranked_sources):
                print('source item')
                print(source_item)
                
                # Get the relevance score
                score = source_item.get('relevance_score', 0)  # Default to 0 if no score
                
                # Keep the first item regardless of score, or items with score >= 7
                if i == 0 and score < 8:
                    low_confidence_flag = True
                    #now intergrate this to have an output row in the google sheet
                    print(f"WARNING: Most relevant source has low confidence (score: {score})")
                if i == 0 or score >= 7:
                    source_id = source_item.get('source')
                    if source_id in content_url_map:
                        ordered_urls.append(content_url_map[source_id]['url'])
                else:
                    print(f"Dropping source with relevance_score {score} (below threshold)")

            print(ordered_urls)
            result['sources'] = ordered_urls

            # If we successfully ordered all URLs, use the ordered list
            """if len(ordered_urls) == len(urls):
                result['sources'] = ordered_urls
                print('ordered URLs')
                print(ordered_urls)
                print('original URLs')
                print(urls)

            else:
                # Fall back to the original URLs if something went wrong
                result['sources'] = urls
                print('original URLs')"""
        else:
            # If no source_relevance in response, use original URLs
            result['sources'] = urls
            print('No order URLs')
            low_confidence_flag = False
        
        if 'source_relevance' in result:
            del result['source_relevance']
        
        result['Low Confidence Flag'] = str(low_confidence_flag)


        state.update({self.output[0]: result})
        return state
    
    #test