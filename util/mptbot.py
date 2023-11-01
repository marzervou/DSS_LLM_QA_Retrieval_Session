"""Wrapper around HuggingFace Pipeline APIs."""
import importlib.util
import logging
from typing import Any, List, Mapping, Optional

from pydantic import Extra

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

DEFAULT_MODEL_ID = "mosaicml/mpt-30b-chat"
DEFAULT_TASK = "text-generation"
VALID_TASKS = ("text-generation")
IP = 'localhost'
PORT = '8880'



"""Wrapper around Text Generation Inference APIs."""

class TGILocalPipeline(LLM):
    pipeline: Any  #: :meta private:
    model_id: str = DEFAULT_MODEL_ID
    """Model name to use."""
    IP: str = IP
    PORT: str = PORT
    model_kwargs: Optional[dict] = None
    """Key word arguments passed to the model."""
    pipeline_kwargs: Optional[dict] = None
    """Key word arguments passed to the pipeline."""
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @classmethod
    def from_model_id(
        cls,
        model_id: str = model_id,
        IP: str = IP,
        PORT: str = PORT,
        trust_remote_code: bool = True,
        model_kwargs: Optional[dict] = None,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> LLM:
        """Construct the pipeline object from model_id and task."""
        try:
          from text_generation import Client

        except ImportError:
            raise ValueError(
                "Could not import text_generation python package. "
                "Please install it with `pip install text_generation`."
            )

        client = Client(f"http://{IP}:{PORT}",timeout=120)

        _pipeline_kwargs = pipeline_kwargs or {}

        _model_kwargs = model_kwargs or {}

        return cls(
            pipeline=client,
            model_id=model_id,
            IP = IP,
            PORT = PORT,
            model_kwargs=_model_kwargs,
            pipeline_kwargs=_pipeline_kwargs,
            **kwargs,
        )

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.pipeline_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_tgi_pipeline"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
      
        generated_text = self.tgi_instruct_generate(prompt)

        return generated_text
    
      
    def tgi_instruct_generate(self,prompt):
        '''
        Def TGI GENERATE
        '''
        if 'max_new_tokens' not in self.pipeline_kwargs:
          self.pipeline_kwargs['max_new_tokens'] = 256

        if 'temperature' not in self.pipeline_kwargs:
          self.pipeline_kwargs['temperature'] = 0.15

        if 'top_k' not in self.pipeline_kwargs:
          self.pipeline_kwargs['top_k'] = 50

        if 'top_p' not in self.pipeline_kwargs:
          self.pipeline_kwargs['top_p'] = 0.6
        
        if 'do_sample' not in self.pipeline_kwargs:
          self.pipeline_kwargs['do_sample'] = False
      

        generator = self.pipeline.generate(prompt,
                            max_new_tokens = self.pipeline_kwargs['max_new_tokens'],
                            temperature=self.pipeline_kwargs['temperature'], 
                            top_p=self.pipeline_kwargs['top_p'],
                            top_k=self.pipeline_kwargs['top_k'],
                            do_sample =  self.pipeline_kwargs['do_sample'])
          
        return generator.generated_text