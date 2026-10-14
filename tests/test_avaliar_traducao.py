"""Indicadores de scripts/avaliar_traducao.py, um caso por defeito."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import avaliar_traducao as av  # noqa: E402

EN = "this is a long enough original sentence"


def test_html():
    assert av.avaliar("isso é <br> um texto", EN)["html"]
    assert av.avaliar("isso &amp; aquilo bem traduzido", EN)["html"]
    assert not av.avaliar("isso é um texto < 3 coisas", EN)["html"]
    assert not av.avaliar("<user> isso é demais", "<user> this is too much for me")["html"]  # ja no original


def test_codificacao():
    assert av.avaliar("nÃ£o sei o que Ã© isso aqui", EN)["codificacao"]
    assert av.avaliar("texto com � no meio da frase", EN)["codificacao"]
    assert not av.avaliar("não sei o que é isso aqui", EN)["codificacao"]


def test_alfabeto_so_o_que_nao_estava_no_original():
    assert av.avaliar("eles falam مرحبا o tempo todo", EN)["alfabeto"]
    assert not av.avaliar("eles falam مرحبا o tempo todo", "they say مرحبا all the time")["alfabeto"]
    assert not av.avaliar("que absurdo 😂😂 mesmo", EN)["alfabeto"]  # emoji nao e letra


def test_ingles_que_sobrou_sem_nome_proprio():
    r = av.avaliar("ateus são totally insignificantes", "atheists are totally insignificant")
    assert r["ingles"] and r["ingles_palavras"] == "totally"
    assert not av.avaliar("o time do Love venceu", "Love's team won the game today")["ingles"]


def test_vazio_igual_tamanho():
    assert av.avaliar("", EN)["vazio"] and av.avaliar("", EN)["quebrada"]
    assert av.avaliar(EN, EN)["igual"]
    assert av.avaliar("curto", EN)["tamanho"]
    assert not av.avaliar("isto é uma frase original longa o bastante", EN)["quebrada"]


def test_mascara_conta_e_nao_e_defeito():
    r = av.avaliar("sai daqui seu vi4d0 de m3rda, p*rra", EN)
    assert r["mascara"] == 3 and not r["quebrada"]
    assert av.mascaras("c@r@lho que coisa") == ["c@r@lho"]


def test_mascara_ignora_o_que_nao_e_palavrao():
    assert av.mascaras("covid19 e 4ever e r2d2 e mp3") == []
