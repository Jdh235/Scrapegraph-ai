from copy import deepcopy
from typing import List, Optional, Type

from pydantic import BaseModel

from ..nodes import GraphIteratorNode, MergeAnswersNode, SearchInternetNode, GenerateAnswerNode
from ..utils.copy import safe_deepcopy
from .abstract_graph import AbstractGraph
from .base_graph import BaseGraph
from .smart_scraper_graph import SmartScraperGraph


class SearchGraphCategorical(AbstractGraph):
    """
    Extended SearchGraph to include GenerateAnswerNode for post-processing.
    """

    def __init__(
        self,
        prompt: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
        user_input: Optional[str] = None,
        document: Optional[List] = None,
        article_description: Optional[str] = None
    ):
        self.max_results = config.get("max_results", 3)
        self.copy_config = safe_deepcopy(config)
        self.copy_schema = deepcopy(schema)
        self.considered_urls = []
        self.user_input = user_input
        self.document = document
        self.article_description = article_description
        super().__init__(prompt, config, schema)

    def _create_graph(self) -> BaseGraph:
        # Create existing nodes
        search_internet_node = SearchInternetNode(
            input="user_prompt",
            output=["urls"],
            node_config={
                "llm_model": self.llm_model,
                "max_results": self.max_results,
                "loader_kwargs": self.loader_kwargs,
                "storage_state": self.copy_config.get("storage_state"),
                "search_engine": self.copy_config.get("search_engine"),
                "serper_api_key": self.copy_config.get("serper_api_key"),
            },
        )

        graph_iterator_node = GraphIteratorNode(
            input="user_prompt & urls",
            output=["results"],
            node_config={
                "graph_instance": SmartScraperGraph,
                "scraper_config": self.copy_config,
            },
            schema=self.copy_schema,
        )

        merge_answers_node = MergeAnswersNode(
            input="user_prompt & results",
            output=["answer"],
            node_config={"llm_model": self.llm_model, "schema": self.copy_schema},
        )

        # Create GenerateAnswerNode
        generate_answer_node = GenerateAnswerNode(
            input="answer & user_input",
            output=["final_answer"],
            node_config={
                "llm_model": self.llm_model,
                "verbose": True,
            }
        )

        # Build the graph with the new node
        return BaseGraph(
            nodes=[search_internet_node, graph_iterator_node, merge_answers_node, generate_answer_node],
            edges=[
                (search_internet_node, graph_iterator_node),
                (graph_iterator_node, merge_answers_node),
                (merge_answers_node, generate_answer_node)
            ],
            entry_point=search_internet_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> str:
        """Run the graph with additional user_input, document, and article_description in the state."""
        inputs = {
            "user_prompt": self.prompt,
            "user_input": self.user_input,
            "article_description": self.article_description or "", #new article_description
        }

        print(inputs)
        self.final_state, self.execution_info = self.graph.execute(inputs)

        if "urls" in self.final_state:
            self.considered_urls = self.final_state["urls"]

        print('final answer')
        print(self.final_state.get("final_answer", "No answer found."))

        return self.final_state.get("final_answer", "No answer found.")

    def get_considered_urls(self) -> List[str]:
        return self.considered_urls