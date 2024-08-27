def calcular_financiamiento(enganche: float, presupuesto: float, tasa: float, plazo: int):
    # Input validations
    if enganche <= 0 or presupuesto <= 0:
        raise ValueError("Enganche o presupuesto inválidos. Verifica los valores ingresados.")
    
    if tasa != 10:
        raise ValueError("La tasa de interés debe ser del 10 %.")
    
    if plazo < 3 or plazo > 6:
        raise ValueError("El plazo debe ser mayor a 3 años y menor a 6.")
    
    if enganche < 0.2 * presupuesto:
        raise ValueError("El enganche debe ser al menos el 20 % del presupuesto.")
    
    if enganche >= presupuesto:
        raise ValueError("El enganche no puede ser mayor o igual al presupuesto.")

    # Calculations
    monto_financiar = presupuesto - enganche
    tasa_mensual = tasa / 12 / 100
    num_pagos = plazo * 12
    
    # Monthly payment calculation using the formula for fixed-rate loans
    pago_mensual = (monto_financiar * tasa_mensual * (1 + tasa_mensual) ** num_pagos) / ((1 + tasa_mensual) ** num_pagos - 1)
    
    # Total amount paid
    total_pagado = pago_mensual * num_pagos + enganche
    
    # Maximum car price affordable with this plan
    precio_maximo_auto = monto_financiar + enganche

    return {
        "monto_financiar": round(monto_financiar, 2),
        "pago_mensual": round(pago_mensual, 2),
        "total_pagado": round(total_pagado, 2),
        "precio_maximo_auto": round(precio_maximo_auto, 2),
        "tasa_anual": tasa,
        "plazo_anos": plazo,
        "num_pagos": num_pagos,
        "enganche": enganche,
        "porcentaje_enganche": round((enganche / presupuesto) * 100, 2)
    }

print(calcular_financiamiento(100000, 500000, 10, 6))