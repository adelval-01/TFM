import asyncio

from livekit import api
from livekit.protocol.sip import CreateSIPOutboundTrunkRequest, SIPOutboundTrunkInfo,  ListSIPOutboundTrunkRequest, DeleteSIPTrunkRequest

async def main():
  livekit_api = api.LiveKitAPI(
    url="wss://test-tfm-3lii83j0.livekit.cloud",
    api_key="APIaRjG6ptSKLTy",
    api_secret="VAovqneYo91unfwHFCGmlJgevds1CZDUIbeufPUuThoE"
  )

  trunk = SIPOutboundTrunkInfo(
    name = "Outbound trunk",
    address = "livekitaitortfm.pstn.twilio.com",
    numbers = ['+12025688661'],
    auth_username = "livekit_TFM",
    auth_password = "Prueba123456"
  )

#   request = CreateSIPOutboundTrunkRequest(
#     trunk = trunk
#   )

#   trunk = await livekit_api.sip.create_sip_outbound_trunk(request)

  print(f"Successfully created {trunk}")


  rules = await livekit_api.sip.list_sip_outbound_trunk(
    ListSIPOutboundTrunkRequest()
  )
  print(f"{rules}")

  
#   sip_trunk_id_list = ["ST_vPrHabYr3UHZ", "ST_XWmfvttSkhKQ", "ST_dsFqugasuDFp", "ST_g7VQuVWxBjZV", "ST_gMziieDaCYJW", "ST_pZDWzhG6ETdR", "ST_rJFQNqqykcb2", "ST_y9tRPnyB3QAz"]
#   for id_sip_trunk in sip_trunk_id_list:
#     request = DeleteSIPTrunkRequest(
#         sip_trunk_id = id_sip_trunk
#     )
#     print(f"Removed SIP inbound trunk {id_sip_trunk}")
#     trunk = await livekit_api.sip.delete_sip_trunk(request)

#   rules = await livekit_api.sip.list_sip_outbound_trunk(
#     ListSIPOutboundTrunkRequest()
#   )
#   print(f"{rules}")

  await livekit_api.aclose()

asyncio.run(main())