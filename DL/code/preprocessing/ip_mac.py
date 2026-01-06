#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

# ip, mac merge
def merge_ip_mac(ip:str, mac:str) -> str:
    if ip is None or mac is None:
        return None
    ip_mac = f"{ip}|{mac}"
    return str(ip_mac)

def get_ip_mac_id_from_vocab(vocab: Dict[str, int], mac: Any, ip: Any) -> int:
    token = merge_ip_mac(ip, mac)
    if not token:
        return -1
    return int(vocab.get(token, -1))