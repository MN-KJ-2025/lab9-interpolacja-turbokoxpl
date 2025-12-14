# =================================  TESTY  ===================================
# Testy do tego pliku obejmują jedynie weryfikację poprawności wyników dla
# prawidłowych danych wejściowych - obsługa niepoprawych danych wejściowych
# nie jest ani wymagana ani sprawdzana. W razie potrzeby lub chęci można ją 
# wykonać w dowolny sposób we własnym zakresie.
# =============================================================================
import numpy as np
from typing import Union
import math

def chebyshev_nodes(n: int = 10) -> Union[np.ndarray ,None]:
    """Funkcja generująca wektor węzłów Czebyszewa drugiego rodzaju (n,) 
    i sortująca wynik od najmniejszego do największego węzła.

    Args:
        n (int): Liczba węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 0:
            return None
        
    if n == 0:
        return np.array([])
        
    if n == 1:
        return np.array([0])
    
    kat = np.linspace(0, np.pi, n)
        
    xk = np.cos(kat)

    return xk



def bar_cheb_weights(n: int = 10) -> Union[np.ndarray ,None]:
    """Funkcja tworząca wektor wag dla węzłów Czebyszewa wymiaru (n,).

    Args:
        n (int): Liczba wag węzłów Czebyszewa.
    
    Returns:
        (np.ndarray): Wektor wag dla węzłów Czebyszewa (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(n, int) or n < 0:
        return None
    if n == 0:
        return np.array([])
    delta=np.ones(n)
    delta[0]=0.5
    delta[-1]=0.5

    wektor = (-1)**np.arange(n)

    return delta*wektor


def barycentric_inte(
    xi: np.ndarray, yi: np.ndarray, wi: np.ndarray, x: np.ndarray
) -> Union[np.ndarray ,None]:
    """Funkcja przeprowadza interpolację metodą barycentryczną dla zadanych 
    węzłów xi i wartości funkcji interpolowanej yi używając wag wi. Zwraca 
    wyliczone wartości funkcji interpolującej dla argumentów x w postaci 
    wektora (n,).

    Args:
        xi (np.ndarray): Wektor węzłów interpolacji (m,).
        yi (np.ndarray): Wektor wartości funkcji interpolowanej w węzłach (m,).
        wi (np.ndarray): Wektor wag interpolacji (m,).
        x (np.ndarray): Wektor argumentów dla funkcji interpolującej (n,).
    
    Returns:
        (np.ndarray): Wektor wartości funkcji interpolującej (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(xi, np.ndarray) or not isinstance(yi, np.ndarray) or not isinstance(wi, np.ndarray) or not isinstance(x, np.ndarray):
        return None
    
    if xi.shape!=yi.shape or xi.shape!=wi.shape or len(xi.shape) != 1 or len(x.shape) != 1:
        return None
    

    roznica=x[:, None] - xi[None, :]

    zero = np.isclose(roznica, 0)

    safe_diff = roznica.copy()
    safe_diff[zero] = 1


    k = wi / safe_diff
    

    licznik = np.sum(k *yi,axis=1)
    
    mianownik = np.sum(k,axis=1)
    
    wynik = licznik/mianownik


    wiersze,koluny = np.where(zero)
    
    if len(wiersze) > 0:
        wynik[wiersze]=yi[koluny]

    return wynik


def L_inf(
    xr: Union[int , float , list, np.ndarray], x: Union[int , float , list , np.ndarray]
) -> Union[float ,None]:
    """Funkcja obliczająca normę L-nieskończoność. Powinna działać zarówno na 
    wartościach skalarnych, listach, jak i wektorach biblioteki numpy.

    Args:
        xr (int | float | list | np.ndarray): Wartość dokładna w postaci 
            skalara, listy lub wektora (n,).
        x (int | float | list | np.ndarray): Wartość przybliżona w postaci 
            skalara, listy lub wektora (n,).

    Returns:
        (float): Wartość normy L-nieskończoność.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(xr, (int, float, list, np.ndarray)) or not isinstance(x,(int, float, list, np.ndarray)):
        return None

    xr_vec = np.array(xr)
    x_vec = np.array(x)

    if not np.issubdtype(xr_vec.dtype, np.number) or not np.issubdtype(x_vec.dtype, np.number):
        return None

    if xr_vec.shape!=x_vec.shape:
        return None

    return float(np.max(np.abs(xr_vec-x_vec)))



def f1(x: Union[int , float , np.ndarray]) -> Union[int,float , np.ndarray]:

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.sign(x)*x+x**2

def f2(x: Union[int , float , np.ndarray]) -> Union[int,float , np.ndarray]:

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.sign(x)*x + x**2


def f3(x: Union[int , float , np.ndarray]) -> Union[int,float , np.ndarray]:

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.abs(np.sin(5*x)**3)

def f4(x: Union[int , float , np.ndarray]) -> Union[int,float , np.ndarray]:

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )
    a=[1,25,100]
    return 1/(1+a*x**2)

def f5(x: Union[int , float , np.ndarray]) -> Union[int,float , np.ndarray]:

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError(
            f"Argument `x` musi być typu `np.ndarray`, `float` lub `int`, otrzymano: {type(x).__name__}."
        )

    return np.sign(x)